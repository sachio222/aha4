"""Tests for the brain-region modules in model/modules.py.

Each test that starts with a bug number reproduces one bug from the review
of the original code. They failed before the fix and pass after it.
"""

import pytest
import torch

from model import modules


# Bug 2: the "hopfield" mask turned every zero into +1, so almost every unit
# came out active instead of just the top k.
def test_bug2_hopfield_mask_marks_only_top_k_as_plus_one():
    x = torch.tensor([[0.1, 0.5, 0.3, 0.2, 0.0]])
    out = modules.get_top_k(x, k=2, mask_type="hopfield", topk_dim=1, scatter_dim=1)
    assert out.tolist() == [[-1.0, 1.0, 1.0, -1.0, -1.0]]


def test_binary_mask_marks_top_k_as_one():
    x = torch.tensor([[0.1, 0.5, 0.3, 0.2, 0.0]])
    out = modules.get_top_k(x, k=2, mask_type="binary", topk_dim=1, scatter_dim=1)
    assert out.tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0]]


# Bug 1: the pseudo-inverse rule used elementwise * where the formula needs a
# matrix product, so stored patterns were not fixed points of the network.
def test_bug1_pinverse_rule_stores_patterns_as_fixed_points():
    torch.manual_seed(0)
    X = torch.sign(torch.randn(10, 64))
    X[X == 0] = 1.0
    ca3 = modules.CA3(64)
    ca3.train(X, method="pinverse")
    assert torch.allclose(X @ ca3.W, X, atol=1e-4)


# Bug 7d: energy was cast to an integer type, which throws away the fraction
# and makes different states look equally stable.
def test_bug7d_hopfield_energy_keeps_fractional_values():
    ca3 = modules.CA3(3)
    ca3.W = torch.tensor([[0.0, 0.5, 0.2], [0.5, 0.0, 0.1], [0.2, 0.1, 0.0]])
    state = torch.tensor([[0.3, -0.7, 0.9]])
    expected = -0.5 * (state @ ca3.W @ state.T).item()
    energy = ca3.energy(state)
    assert energy.is_floating_point()
    assert energy.flatten()[0].item() == pytest.approx(expected)


# Bug 3: top-k sparsity ran across the batch (dim 0), so one drawing's code
# depended on whatever else was in the batch. The paper picks the top k
# filters at each position of each sample.
def test_bug3_pretrain_vc_code_does_not_depend_on_the_batch():
    torch.manual_seed(0)
    vc = modules.ECPretrain(1, 121, 10, 5, 4).eval()
    x = torch.rand(4, 1, 52, 52)
    with torch.no_grad():
        batch = vc(x, k=1)
        single = vc(x[:1], k=1)
    assert torch.allclose(batch[0], single[0], atol=1e-6)


def test_bug3_pretrain_vc_code_is_k_sparse_over_filters():
    torch.manual_seed(0)
    vc = modules.ECPretrain(1, 121, 10, 5, 4).eval()
    with torch.no_grad():
        code = vc.encode(torch.rand(4, 1, 52, 52), k=2)
    assert code.shape == (4, 121, 11, 11)
    active_per_position = (code != 0).sum(dim=1)
    assert active_per_position.max().item() <= 2


def test_bug3_inference_vc_code_does_not_depend_on_the_batch():
    torch.manual_seed(0)
    vc = modules.EC(interest_filter=False).eval()
    x = torch.rand(4, 1, 52, 52)
    with torch.no_grad():
        batch = vc(x, k=4)
        single = vc(x[:1], k=4)
    assert torch.allclose(batch[0], single[0], atol=1e-6)


# Bug 7a: reset() rebuilt the mask as torch.ones(10), the wrong shape.
def test_bug7a_inhibition_mask_reset_restores_full_shape():
    mask = modules.InhibitionMask(dim=225)
    fired = torch.zeros(1, 225)
    fired[0, :10] = 1.0
    mask.update(fired)
    assert (mask()[0, :10] < 1).all()
    mask.reset()
    assert mask().shape == (1, 225)
    assert torch.all(mask() == 1)


def test_dg_gives_k_active_units_and_separates_repeated_inputs():
    torch.manual_seed(0)
    dg = modules.DG(D_in=50, D_out=225, k=10)
    x = torch.rand(1, 50).repeat(20, 1)
    with torch.no_grad():
        out = dg(x)
    assert torch.all(out.sum(dim=1) == 10)
    # The same drawing shown twice in a row must not reuse the same units,
    # because units that just fired are inhibited.
    assert (out[0] * out[1]).sum().item() == 0


def test_hopfield_recall_completes_a_corrupted_pattern():
    torch.manual_seed(0)
    X = -torch.ones(20, 225)
    for i in range(20):
        X[i, torch.randperm(225)[:10]] = 1.0
    ca3 = modules.CA3(225)
    ca3.train(X)
    cue = X.clone()
    cue[:, :20] *= -1  # flip some units in every pattern
    out = ca3.update(cue)
    assert torch.all(torch.sign(out) == X)


DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_hopfield_trains_and_recalls_on_every_available_device(device):
    torch.manual_seed(0)
    X = -torch.ones(5, 225)
    for i in range(5):
        X[i, torch.randperm(225)[:10]] = 1.0
    X = X.to(device)
    ca3 = modules.CA3(225)
    ca3.train(X)
    out = ca3.update(X)
    assert out.device.type == device
    assert torch.all(torch.sign(out) == X)
