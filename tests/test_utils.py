"""Tests for checkpoint and device helpers in utils/utils.py."""

import pytest
import torch

from utils import utils


# Bug 7b: load_checkpoint did `raise ("File does not exist...")`, which raises
# a TypeError about strings instead of telling you the file is missing. It
# also checked the folder, not the file.
def test_bug7b_missing_checkpoint_folder_raises_file_not_found(tmp_path):
    model = torch.nn.Linear(2, 2)
    with pytest.raises(FileNotFoundError):
        utils.load_checkpoint(tmp_path / "nope", model, name="missing")


def test_bug7b_missing_checkpoint_file_raises_file_not_found(tmp_path):
    model = torch.nn.Linear(2, 2)
    with pytest.raises(FileNotFoundError):
        utils.load_checkpoint(tmp_path, model, name="missing")


# Bug 7c: the CA1 checkpoint was saved with the PR optimizer. A save state
# should refuse an optimizer that does not own the model's parameters.
def test_bug7c_save_state_rejects_optimizer_from_another_model():
    a = torch.nn.Linear(2, 2)
    b = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError):
        utils.get_save_state(0, a, torch.optim.Adam(b.parameters()))
    state = utils.get_save_state(0, a, torch.optim.Adam(a.parameters()))
    assert set(state) >= {"epoch", "state_dict", "optim_dict"}


def test_checkpoint_round_trip(tmp_path):
    a = torch.nn.Linear(2, 2)
    utils.save_checkpoint({"state_dict": a.state_dict()}, tmp_path, name="lin")
    b = torch.nn.Linear(2, 2)
    utils.load_checkpoint(tmp_path, b, name="lin")
    assert torch.equal(a.weight, b.weight)


# Bug 5: every GPU path did `x, _ = x.cuda(...)`, which unpacks the batch
# itself. These helpers pick a device and move a whole batch to it.
def test_bug5_resolve_device_cpu_and_auto():
    assert utils.resolve_device("cpu") == torch.device("cpu")
    assert utils.resolve_device("auto").type in {"cpu", "mps", "cuda"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present here")
def test_bug5_resolve_device_rejects_a_missing_backend():
    with pytest.raises(RuntimeError):
        utils.resolve_device("cuda")


def test_bug5_to_device_keeps_the_whole_batch():
    x = torch.rand(3, 1, 52, 52)
    labels = torch.arange(3)
    out = utils.to_device((x, labels), torch.device("cpu"))
    assert out.shape == (3, 1, 52, 52)
