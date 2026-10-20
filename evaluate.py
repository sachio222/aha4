"""Scores AHA on the one-shot Omniglot benchmark, as in the paper.

Runs Lake et al.'s 20 official one-shot runs once per seed. In each run AHA
studies 20 drawings, recalls 20 drawings of the same characters by another
person, and gets scored by minimum-MSE matching at each stage:

    pixels  raw images, a floor to compare against
    vc      the vision part alone, the paper's LTM baseline (71.6%)
    pr      PR(EC-CA3) output, the paper's headline LTM+AHA number (86.4%)
    pc      PC(CA3) after the Hopfield net settles (paper: no better than LTM)
    pm      PM(CA1) image reconstruction

Example:
    python evaluate.py --checkpoint checkpoints/vc_pretrained.pth.tar --seeds 10
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
    p.add_argument("--checkpoint", default="checkpoints/vc_pretrained.pth.tar")
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
    p.add_argument("--out", default="results/oneshot.json")
    return p.parse_args(argv)


def summarise(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def main(argv=None):
    args = parse_args(argv)
    device = utils.resolve_device(args.device)
    utils.set_seed(args.first_seed)

    config = AHAConfig(ca3_rule=args.ca3_rule, study_steps=args.study_steps)
    vc = load_vc(args.checkpoint, interest_filter=not args.no_interest_filter, device=device)
    aha = AHA(vc, config, device=device)
    runs = data.load_runs(args.data_dir, config.image_size, centre=not args.no_centre)

    print("Device: {}. {} runs x {} seeds.".format(device, len(runs), args.seeds))
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    result = oneshot.evaluate(aha, runs, seeds)

    summary = {stage: summarise(acc) for stage, acc in result["accuracy"].items()}
    print("\nOne-shot classification accuracy, mean over {} seeds".format(args.seeds))
    print("{:<8}{:>10}{:>10}{:>10}".format("stage", "ours", "std", "paper"))
    for stage, s in summary.items():
        paper = "{:.1%}".format(PAPER[stage]) if stage in PAPER else "-"
        print("{:<8}{:>10.1%}{:>10.1%}{:>10}".format(stage, s["mean"], s["std"], paper))
    print("Recall-loss (MSE): {:.4f}".format(statistics.mean(result["recall_loss"])))
    print("Took {:.0f} s".format(result["seconds"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args),
        "config": asdict(config),
        "device": str(device),
        "machine": platform.platform(),
        "torch": torch.__version__,
        "paper": PAPER,
        "summary": summary,
        "per_seed": result,
    }, indent=2))
    print("Wrote {}".format(out))


if __name__ == "__main__":
    main()
