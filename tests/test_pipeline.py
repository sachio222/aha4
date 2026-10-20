"""Tests that wire the parts together the way the scripts do."""

import math

import torch

from model import modules
from model.aha import AHA, AHAConfig
from utils import oneshot


# Bug 5 (pretraining side): one pretraining step must use the whole batch.
def test_bug5_pretrain_step_runs_on_a_full_batch():
    import pretrain

    torch.manual_seed(0)
    model = modules.ECPretrain()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = (torch.rand(3, 1, 52, 52), torch.zeros(3))
    loss = pretrain.pretrain_step(model, optimizer, batch, torch.device("cpu"), k=1)
    assert math.isfinite(loss)


# Bug 6: pretraining used 9x9 filters (the paper says 10x10). The paper
# pretrains at stride 5 and runs at stride 1, so the same filters have to fit
# both layouts.
def test_bug6_pretrained_filters_fit_the_inference_vc():
    torch.manual_seed(0)
    pre = modules.ECPretrain()
    assert pre.encoder.weight.shape == (121, 1, 10, 10)
    assert pre.encoder.stride == (5, 5)
    assert pre(torch.rand(2, 1, 52, 52), k=1).shape == (2, 1, 52, 52)

    vc = modules.EC()
    vc.load_pretrained(pre.state_dict())
    assert vc.encoder.stride == (1, 1)
    assert torch.equal(vc.encoder.weight, pre.encoder.weight)
    assert vc.feature_map(torch.rand(2, 1, 52, 52), k=4).shape == (2, 121, 52, 52)


# Bug 4: train.py built DG and PR for 27225 inputs while the vision part puts
# out 121 x 13 x 13 = 20449. The sizes should come from the vision part.
def test_bug4_stm_input_size_matches_vc_output():
    torch.manual_seed(0)
    vc = modules.EC(interest_filter=False)
    assert vc.output_dim(52) == 121 * 13 * 13 == 20449

    aha = AHA(vc, AHAConfig(pr_hidden=64, study_steps=2), device=torch.device("cpu"))
    aha.reset(seed=0)
    assert aha.dg.fc1.in_features == 20449
    assert aha.pr.fc1.in_features == 20449

    aha.study(torch.rand(5, 1, 52, 52))
    recall = aha.recall(torch.rand(5, 1, 52, 52))
    assert recall["pm"].shape == (5, 1, 52, 52)


def test_aha_recognises_the_drawings_it_studied():
    torch.manual_seed(0)
    images = (torch.rand(20, 1, 52, 52) > 0.9).float()
    vc = modules.EC(interest_filter=False)
    aha = AHA(vc, AHAConfig(), device=torch.device("cpu"))
    aha.reset(seed=0)
    study = aha.study(images)
    recall = aha.recall(images)
    labels = torch.arange(20)
    assert oneshot.run_accuracy(study["pr"], recall["pr"], labels) == 1.0
    assert oneshot.run_accuracy(study["pc"], recall["pc"], labels) >= 0.9
