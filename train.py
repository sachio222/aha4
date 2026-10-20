"""Trains AHA's short-term memory on one study set and shows what it recalls.

In AHA the hippocampal parts train inside every evaluation step: they study
a set of drawings once, then get tested, then get wiped (paper Section
2.3). So "training" here means one study-and-recall cycle on one of Lake
et al.'s one-shot runs. The script prints each stage's accuracy on that
run and saves a picture of the signals as they pass through AHA, laid out
like the paper's Figure 7:

    row 1  study drawings
    row 2  PS(DG) patterns stored for them
    row 3  test drawings of the same characters, by another person
    row 4  PR(EC-CA3) output for each test drawing
    row 5  PC(CA3) state after the Hopfield net settles
    row 6  PM(CA1) image recalled from that state

Test drawings are reordered so each column lines up with its study drawing.

Example:
    python train.py --run 1 --device mps
"""

import argparse
from pathlib import Path

import torch

from model.aha import AHA, AHAConfig, load_vc
from utils import data, oneshot, utils


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default="checkpoints/vc_pretrained.pth.tar")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    p.add_argument("--no-centre", action="store_true",
                   help="skip centring each drawing on its centre of ink (ablation)")
    p.add_argument("--run", type=int, default=1, help="which one-shot run, 1 to 20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-interest-filter", action="store_true")
    p.add_argument("--out", default="figures/aha_run01.png")
    return p.parse_args(argv)


def plot_signals(study, recall, labels, path):
    """Saves the six-row picture described at the top of this file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Put each test drawing in the column of the study drawing it matches.
    order = torch.argsort(labels.cpu())
    side = int(study["dg"].shape[1] ** 0.5)
    rows = [
        ("Study", study["pixels"]),
        ("PS(DG)", study["dg"].view(-1, 1, side, side)),
        ("Test", recall["pixels"][order]),
        ("PR(EC-CA3)", recall["pr"][order].view(-1, 1, side, side)),
        ("PC(CA3)", recall["pc"][order].view(-1, 1, side, side)),
        ("PM(CA1)", recall["pm"][order]),
    ]

    n = study["pixels"].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 0.8, len(rows) * 0.9))
    for r, (name, images) in enumerate(rows):
        images = images.detach().cpu()
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(images[c, 0], cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    device = utils.resolve_device(args.device)
    utils.set_seed(args.seed)

    config = AHAConfig()
    vc = load_vc(args.checkpoint, interest_filter=not args.no_interest_filter, device=device)
    aha = AHA(vc, config, device=device)
    run = data.load_runs(args.data_dir, config.image_size, centre=not args.no_centre)[args.run - 1]

    aha.reset(seed=args.seed)
    study = aha.study(run["study"])
    recall = aha.recall(run["recall"])

    print("Run {}, seed {}, device {}".format(args.run, args.seed, device))
    for stage in oneshot.STAGES:
        acc = oneshot.run_accuracy(study[stage], recall[stage], run["labels"])
        print("  {:<7} {:.0%}".format(stage, acc))
    print("  recall-loss {:.4f}".format(
        oneshot.recall_loss(recall["pm"], run["study"], run["labels"])))

    plot_signals(study, recall, run["labels"], args.out)
    print("Wrote {}".format(args.out))


if __name__ == "__main__":
    main()
