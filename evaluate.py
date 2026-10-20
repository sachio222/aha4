"""Scores AHA on the one-shot Omniglot benchmark, as in the paper.

Runs Lake et al.'s 20 official one-shot runs once per seed. In each run AHA
studies 20 drawings, recalls 20 drawings of the same characters by another
person, and gets scored by minimum-MSE matching at each stage:

    pixels  raw images, a floor to compare against
    vc      the vision part alone, the paper's LTM baseline (71.6%)
    pr      PR(EC-CA3) output, the paper's headline LTM+AHA number (86.4%)
    pc      PC(CA3) after the Hopfield net settles (paper: no better than LTM)
    pm      PM(CA1) image reconstruction

Seeds set the hippocampal parts only. The vision part comes from the
checkpoint, so with one checkpoint the vc and pixels scores are the same on
every seed. Pass several checkpoints, pretrained with different seeds, to
measure how much pretraining moves the scores: each one runs the full set
of seeds, and the summary spreads across the checkpoints.

Example:
    python evaluate.py --seeds 10
    python evaluate.py --match test-to-study --out results/match_test_to_study.json
    python evaluate.py --checkpoint checkpoints/vc_s0.pth.tar checkpoints/vc_s1.pth.tar \\
        checkpoints/vc_s2.pth.tar --out results/pretrain_seeds.json
"""

import argparse
import json
import platform
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from model.aha import AHA, AHAConfig, load_vc
from utils import data, oneshot, utils

# One-shot classification accuracy without noise or occlusion, from the
# paper's Section 4.1.1 and Table 1. The paper gives no number for PC(CA3)
# beyond "no better than LTM".
PAPER = {"vc": 0.716, "pr": 0.864}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", nargs="+", default=["checkpoints/vc_pretrained.pth.tar"],
                   help="one or more pretrained vision networks")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    p.add_argument("--no-centre", action="store_true",
                   help="skip centring each drawing on its centre of ink (ablation)")
    p.add_argument("--seeds", type=int, default=10, help="how many seeds to repeat over")
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--no-interest-filter", action="store_true",
                   help="turn off the VC interest filter (ablation)")
    p.add_argument("--ca3-rule", default="pinverse", choices=["pinverse", "hebbian"])
    p.add_argument("--study-steps", type=int, default=60)
    p.add_argument("--match", default=oneshot.DIRECTIONS[0], choices=oneshot.DIRECTIONS,
                   help="study-to-test as the paper describes, or test-to-study")
    p.add_argument("--normalise", action="store_true",
                   help="L2-normalise every stage's vectors before MSE matching")
    p.add_argument("--normalise-input", action="store_true",
                   help="L2-normalise the VC output before it feeds DG and PR")
    p.add_argument("--out", default="results/oneshot.json")
    return p.parse_args(argv)


def summarise(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def pretrain_settings(checkpoint):
    """The pretrain.py arguments saved in a checkpoint. Older files lack
    "loss", and those were all trained with bce."""
    saved = torch.load(checkpoint, map_location="cpu").get("args", {})
    return {"loss": saved.get("loss", "bce"), "seed": saved.get("seed"),
            "no_centre": saved.get("no_centre", False), "batches": saved.get("batches")}


def evaluate_checkpoint(checkpoint, args, config, runs, device):
    vc = load_vc(checkpoint, interest_filter=not args.no_interest_filter, device=device)
    aha = AHA(vc, config, device=device)
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    return oneshot.evaluate(aha, runs, seeds, direction=args.match, normalise=args.normalise)


def main(argv=None):
    args = parse_args(argv)
    device = utils.resolve_device(args.device)
    utils.set_seed(args.first_seed)

    config = AHAConfig(ca3_rule=args.ca3_rule, study_steps=args.study_steps,
                       normalise_input=args.normalise_input)
    runs = data.load_runs(args.data_dir, config.image_size, centre=not args.no_centre)

    print("Device: {}. {} runs x {} seeds x {} checkpoint(s). Matching {}{}.".format(
        device, len(runs), args.seeds, len(args.checkpoint), args.match,
        ", normalised" if args.normalise else ""))

    per_checkpoint = []
    for checkpoint in args.checkpoint:
        print("Checkpoint {}".format(checkpoint))
        result = evaluate_checkpoint(checkpoint, args, config, runs, device)
        per_checkpoint.append({
            "checkpoint": checkpoint,
            "pretrain": pretrain_settings(checkpoint),
            "summary": {stage: summarise(acc) for stage, acc in result["accuracy"].items()},
            "per_seed": result,
        })

    if len(per_checkpoint) == 1:
        # Spread across the hippocampal seeds.
        summary = per_checkpoint[0]["summary"]
        spread = "seeds"
    else:
        # Spread across pretraining checkpoints, each averaged over its seeds.
        summary = {stage: summarise([c["summary"][stage]["mean"] for c in per_checkpoint])
                   for stage in oneshot.STAGES}
        spread = "checkpoints"

    loss = [l for c in per_checkpoint for l in c["per_seed"]["recall_loss"]]
    seconds = sum(c["per_seed"]["seconds"] for c in per_checkpoint)

    print("\nOne-shot classification accuracy, mean over {} seeds{}".format(
        args.seeds, "" if spread == "seeds" else
        " and {} checkpoints (std across checkpoints)".format(len(per_checkpoint))))
    print("{:<8}{:>10}{:>10}{:>10}".format("stage", "ours", "std", "paper"))
    for stage, s in summary.items():
        paper = "{:.1%}".format(PAPER[stage]) if stage in PAPER else "-"
        print("{:<8}{:>10.1%}{:>10.1%}{:>10}".format(stage, s["mean"], s["std"], paper))
    print("Recall-loss (MSE): {:.4f}".format(statistics.mean(loss)))
    print("Took {:.0f} s".format(seconds))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "args": vars(args),
        "config": asdict(config),
        "device": str(device),
        "machine": platform.platform(),
        "torch": torch.__version__,
        "paper": PAPER,
        "spread": spread,
        "pretrain": [c["pretrain"] for c in per_checkpoint],
        "summary": summary,
        "seconds": seconds,
    }
    if len(per_checkpoint) == 1:
        record["per_seed"] = per_checkpoint[0]["per_seed"]
    else:
        record["per_checkpoint"] = per_checkpoint
    out.write_text(json.dumps(record, indent=2))
    print("Wrote {}".format(out))


if __name__ == "__main__":
    main()
