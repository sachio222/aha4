"""Tests for the details the paper leaves open in one-shot scoring and pretraining.

N1: the paper (p8, Section 3.1) presents one "train" character and asks for
the matching character among 20 "test" ones. That is study-to-test: each
study drawing looks for its partner among the test drawings. The first
version of this repo went the other way.

N2: the authors' later code (Cerenaut/pt-aha) divides each VC output by its
L2 norm. The paper doesn't say. Both are options here.
"""

import json
import math

import pytest
import torch
import torch.nn.functional as F

from model import modules
from model.aha import AHA, AHAConfig
from utils import oneshot


# Three study points on a line and three test points. Test items 0 and 1
# both sit nearest study item 0, so test-to-study gets item 1 wrong, while
# each study item still finds its own partner first.
STUDY = torch.tensor([[0.0], [1.0], [10.0]])
TEST = torch.tensor([[0.4], [0.45], [9.0]])
LABELS = torch.tensor([0, 1, 2])


def test_n1_default_matching_is_study_to_test():
    assert oneshot.DIRECTIONS[0] == "study-to-test"
    assert oneshot.run_accuracy(STUDY, TEST, LABELS) == 1.0


def test_n1_test_to_study_is_still_available():
    acc = oneshot.run_accuracy(STUDY, TEST, LABELS, direction="test-to-study")
    assert acc == pytest.approx(2 / 3)


def test_n1_study_to_test_handles_shuffled_labels():
    torch.manual_seed(0)
    study = torch.rand(20, 1, 8, 8)
    order = torch.randperm(20)
    assert oneshot.run_accuracy(study, study[order], order) == 1.0
    wrong = torch.roll(order, 1)
    assert oneshot.run_accuracy(study, study[order], wrong) == 0.0


def test_unknown_direction_is_refused():
    with pytest.raises(ValueError):
        oneshot.run_accuracy(STUDY, TEST, LABELS, direction="sideways")


# Study item 1 is ten times longer than everything else, so at raw scale
# both test items sit nearest study item 0. Pointing the same way is what
# should count once vectors are normalised.
SCALED_STUDY = torch.tensor([[1.0, 0.0], [0.0, 10.0]])
SCALED_TEST = torch.tensor([[5.0, 0.0], [0.0, 1.0]])
SCALED_LABELS = torch.tensor([0, 1])


@pytest.mark.parametrize("direction", ["study-to-test", "test-to-study"])
def test_n2_normalise_matches_on_direction_not_length(direction):
    raw = oneshot.run_accuracy(SCALED_STUDY, SCALED_TEST, SCALED_LABELS, direction=direction)
    norm = oneshot.run_accuracy(SCALED_STUDY, SCALED_TEST, SCALED_LABELS,
                                direction=direction, normalise=True)
    assert raw == 0.5
    assert norm == 1.0


class FakeAHA():
    """Hands back fixed states so evaluate() can be checked without training."""

    def reset(self, seed=0):
        pass

    def study(self, images):
        return {"vc": SCALED_STUDY, "pm": images}

    def recall(self, images):
        return {"vc": SCALED_TEST, "pm": images}


def test_evaluate_passes_direction_and_normalise_through():
    runs = [{"study": torch.zeros(2, 1, 2, 2), "recall": torch.zeros(2, 1, 2, 2),
             "labels": SCALED_LABELS}]
    raw = oneshot.evaluate(FakeAHA(), runs, [0], stages=("vc",), log=lambda s: None)
    norm = oneshot.evaluate(FakeAHA(), runs, [0], stages=("vc",), normalise=True,
                            log=lambda s: None)
    assert raw["accuracy"]["vc"] == [0.5]
    assert norm["accuracy"]["vc"] == [1.0]


def test_evaluate_flags():
    import evaluate

    args = evaluate.parse_args([])
    assert args.match == "study-to-test"
    assert args.normalise is False
    assert args.normalise_input is False
    assert args.checkpoint == ["checkpoints/vc_pretrained.pth.tar"]

    args = evaluate.parse_args(["--match", "test-to-study", "--normalise", "--normalise-input",
                                "--checkpoint", "a.pth.tar", "b.pth.tar", "c.pth.tar"])
    assert args.match == "test-to-study"
    assert args.normalise and args.normalise_input
    assert args.checkpoint == ["a.pth.tar", "b.pth.tar", "c.pth.tar"]

    with pytest.raises(SystemExit):
        evaluate.parse_args(["--match", "sideways"])


def test_n2_normalise_input_feeds_unit_length_vision_codes_to_the_stm():
    torch.manual_seed(0)
    vc = modules.EC(interest_filter=False)
    images = (torch.rand(4, 1, 52, 52) > 0.9).float()

    plain = AHA(vc, AHAConfig(pr_hidden=16, study_steps=1), device=torch.device("cpu"))
    normed = AHA(vc, AHAConfig(pr_hidden=16, study_steps=1, normalise_input=True),
                 device=torch.device("cpu"))
    assert not torch.allclose(plain.encode(images).norm(dim=1), torch.ones(4))
    assert torch.allclose(normed.encode(images).norm(dim=1), torch.ones(4), atol=1e-5)

    normed.reset(seed=0)
    study = normed.study(images)
    assert torch.allclose(study["vc"].norm(dim=1), torch.ones(4), atol=1e-5)


@pytest.mark.parametrize("loss", ["bce", "mse"])
def test_pretrain_loss_option(loss):
    import pretrain

    assert pretrain.parse_args([]).loss == "bce"
    assert pretrain.parse_args(["--loss", "mse"]).loss == "mse"

    torch.manual_seed(0)
    model = modules.ECPretrain()
    x = torch.rand(3, 1, 52, 52)
    with torch.no_grad():
        logits = model.logits(x, k=1)
    expected = (F.mse_loss(logits, x) if loss == "mse"
                else F.binary_cross_entropy_with_logits(logits, x)).item()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    got = pretrain.pretrain_step(model, optimizer, (x, torch.zeros(3)), torch.device("cpu"),
                                 k=1, loss=loss)
    assert math.isfinite(got)
    assert got == pytest.approx(expected, rel=1e-5)


def test_make_tables_reads_every_row_from_json(tmp_path):
    from scripts import make_tables

    def write(name, **summary):
        (tmp_path / name).write_text(json.dumps({
            "args": {"match": "study-to-test", "normalise": False},
            "summary": {k: {"mean": v, "std": 0.01, "min": v, "max": v}
                        for k, v in summary.items()},
        }))

    write("oneshot.json", pixels=0.4, vc=0.5, pr=0.6, pc=0.61, pm=0.62)
    write("match_test_to_study.json", pixels=0.4, vc=0.55, pr=0.65, pc=0.66, pm=0.67)
    rows = make_tables.load_rows(tmp_path)
    names = [r["file"] for r in rows]
    assert names[:2] == ["oneshot.json", "match_test_to_study.json"]

    md = make_tables.grid_table(rows)
    assert "60.0%" in md and "65.0%" in md

    csv_path = tmp_path / "grid.csv"
    make_tables.write_csv(rows, csv_path)
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("file,")
    assert len(lines) == 3
