"""Scoring for the one-shot Omniglot classification benchmark.

The benchmark comes from Lake, Salakhutdinov and Tenenbaum (2015). There
are 20 runs. Each run has 20 study drawings, one per character, all from
one alphabet and one person, and 20 test drawings of the same characters
by a different person. A model gets one look at the study set, then has
to say which study character each test drawing shows. Chance is 1 in 20.

AHA's paper (Section 3.1) matches by minimum mean squared error (MSE)
between an internal state at study time and the same state at recall time,
and averages accuracy over the 20 runs. That's what this file does.
"""

import time

import torch

# Stages of the model you can score, in the order a drawing passes them.
STAGES = ("pixels", "vc", "pr", "pc", "pm")


def min_mse_match(study, recall):
    """For each recall item, the index of the study item with lowest MSE."""
    s = study.detach().flatten(1).float().cpu()
    r = recall.detach().flatten(1).float().cpu()
    mse = torch.cdist(r, s) ** 2 / s.shape[1]
    return mse.argmin(dim=1)


def run_accuracy(study, recall, labels):
    """Share of recall items matched to their true study item."""
    return (min_mse_match(study, recall) == labels.cpu()).float().mean().item()


def recall_loss(reconstruction, study_images, labels):
    """MSE between each recalled image and the study drawing it should be.

    Paper Section 3.3 calls this recall-loss.
    """
    target = study_images.cpu()[labels.cpu()]
    return torch.mean((reconstruction.detach().cpu() - target) ** 2).item()


def evaluate(aha, runs, seeds, stages=STAGES, log=print):
    """Runs every one-shot run once per seed and scores each stage.

    The STM is reset before every run with a seed made from the seed and
    run number, so each (seed, run) pair repeats exactly on CPU.

    Returns a dict with per-seed mean accuracy for each stage, per-seed mean
    recall-loss, and the wall time.
    """
    start = time.time()
    acc = {stage: [] for stage in stages}
    loss = []

    for seed in seeds:
        per_run = {stage: [] for stage in stages}
        per_run_loss = []
        for i, run in enumerate(runs):
            aha.reset(seed=seed * 1000 + i)
            study = aha.study(run["study"])
            recall = aha.recall(run["recall"])
            for stage in stages:
                per_run[stage].append(run_accuracy(study[stage], recall[stage], run["labels"]))
            per_run_loss.append(recall_loss(recall["pm"], run["study"], run["labels"]))

        for stage in stages:
            acc[stage].append(sum(per_run[stage]) / len(runs))
        loss.append(sum(per_run_loss) / len(runs))
        log("seed {:>3}  ".format(seed) + "  ".join(
            "{} {:.1%}".format(stage, acc[stage][-1]) for stage in stages))

    return {"accuracy": acc, "recall_loss": loss, "seconds": time.time() - start}
