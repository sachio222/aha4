"""Tests for the one-shot Omniglot scoring.

Bug 8: the old "test" paired each drawing with the next dataset index and
was judged by eye. The real benchmark (Lake et al. 2015) gives 20 runs. In
each run you study 20 drawings, then match 20 new drawings of the same
characters by a different person back to the study set.
"""

import torch

from utils import data, oneshot


def test_bug8_min_mse_match_picks_the_nearest_study_item():
    study = torch.tensor([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    recall = torch.tensor([[4.0, 4.5], [0.2, -0.1]])
    assert oneshot.min_mse_match(study, recall).tolist() == [2, 0]


def test_bug8_identical_sets_in_shuffled_order_score_perfectly():
    torch.manual_seed(0)
    study = torch.rand(20, 1, 8, 8)
    order = torch.randperm(20)
    recall = study[order]
    assert oneshot.run_accuracy(study, recall, order) == 1.0


def test_bug8_wrong_labels_score_zero():
    study = torch.eye(4)
    labels = torch.tensor([1, 2, 3, 0])
    assert oneshot.run_accuracy(study, study, labels) == 0.0


def test_bug8_parse_class_labels_maps_test_items_to_training_classes():
    lines = [
        "run01/test/item01.png run01/training/class08.png",
        "run01/test/item02.png run01/training/class09.png",
        "",
    ]
    assert data.parse_class_labels(lines) == [7, 8]
