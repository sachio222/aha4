"""Scoring for the one-shot Omniglot classification benchmark.

The benchmark comes from Lake, Salakhutdinov and Tenenbaum (2015). There
are 20 runs. Each run has 20 study drawings, one per character, all from
one alphabet and one person, and 20 test drawings of the same characters
by a different person. A model gets one look at the study set. Chance is
1 in 20.

AHA's paper (Section 3.1) matches by minimum mean squared error (MSE)
between an internal state at study time and the same state at recall time,
and averages accuracy over the 20 runs. That's what this file does.

Two details the paper leaves open, or that the first version of this repo
got wrong, are options here:

    direction   The paper (p8) presents one "train" character and asks for
                the matching character among a "test" set of 20. So each
                study drawing looks for its partner among the test drawings:
                study-to-test, the default. test-to-study asks the opposite
                question: which study drawing each test drawing is closest
                to. The two can disagree, because two test drawings can
                share a nearest study drawing.
    normalise   Divide every vector by its L2 norm before taking the MSE,
                so matching compares direction and ignores length. The
                authors' later code (Cerenaut/pt-aha) does this to the
                vision output. The paper doesn't mention it.
"""

import time

import torch
import torch.nn.functional as F

# Stages of the model you can score, in the order a drawing passes them.
STAGES = ("pixels", "vc", "pr", "pc", "pm")

# Matching directions. The first is the default.
DIRECTIONS = ("study-to-test", "test-to-study")


def _flat(x, normalise=False):
    x = x.detach().flatten(1).float().cpu()
    return F.normalize(x, dim=1) if normalise else x


def mse_matrix(study, recall, normalise=False):
    """MSE between every recall item (rows) and every study item (columns)."""
    s, r = _flat(study, normalise), _flat(recall, normalise)
    return torch.cdist(r, s) ** 2 / s.shape[1]


def min_mse_match(study, recall, normalise=False):
    """For each recall item, the index of the study item with lowest MSE."""
    return mse_matrix(study, recall, normalise).argmin(dim=1)


def run_accuracy(study, recall, labels, direction=DIRECTIONS[0], normalise=False):
    """Share of items matched to their true partner.

    labels[j] is the study index that recall item j shows.

    study-to-test: each study item picks its nearest recall item, and is
    right when that recall item shows it. test-to-study: each recall item
    picks its nearest study item, and is right when that's its label.
    """
    mse = mse_matrix(study, recall, normalise)
    labels = labels.cpu()
    if direction == "study-to-test":
        nearest = mse.argmin(dim=0)
        hits = labels[nearest] == torch.arange(mse.shape[1])
    elif direction == "test-to-study":
        hits = mse.argmin(dim=1) == labels
    else:
        raise ValueError("direction must be one of {}".format(DIRECTIONS))
    return hits.float().mean().item()


def recall_loss(reconstruction, study_images, labels):
    """MSE between each recalled image and the study drawing it should be.

    Paper Section 3.3 calls this recall-loss.
    """
    target = study_images.cpu()[labels.cpu()]
    return torch.mean((reconstruction.detach().cpu() - target) ** 2).item()


def evaluate(aha, runs, seeds, stages=STAGES, direction=DIRECTIONS[0], normalise=False,
             log=print):
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
                per_run[stage].append(run_accuracy(study[stage], recall[stage], run["labels"],
                                                   direction, normalise))
            per_run_loss.append(recall_loss(recall["pm"], run["study"], run["labels"]))

        for stage in stages:
            acc[stage].append(sum(per_run[stage]) / len(runs))
        loss.append(sum(per_run_loss) / len(runs))
        log("seed {:>3}  ".format(seed) + "  ".join(
            "{} {:.1%}".format(stage, acc[stage][-1]) for stage in stages))

    return {"accuracy": acc, "recall_loss": loss, "seconds": time.time() - start}
