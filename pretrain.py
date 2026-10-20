"""Pretrains the vision part of AHA, VC(EC), on the Omniglot background set.

This is stage 1 of the paper's framework (Section 2.3): the long-term memory
learns general visual features from many drawings over many batches. The
filters it ends up with look like pen strokes. The hippocampal parts never
see these 30 alphabets.

Paper settings (Table 3): 121 filters of 10x10, stride 5, one winner per
position, learning rate 0.001, 2000 batches of 128.

Example:
    python pretrain.py --device mps
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import modules
from utils import data, utils


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    p.add_argument("--no-centre", action="store_true",
                   help="skip centring each drawing on its centre of ink (ablation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batches", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=1, help="winning filters per position")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--name", default="vc_pretrained")
    p.add_argument("--filters-plot", default="figures/vc_filters.png")
    return p.parse_args(argv)


def pretrain_step(model, optimizer, batch, device, k=1):
    """One autoencoder update on a full batch. Returns the loss."""
    x = utils.to_device(batch, device)
    loss = F.binary_cross_entropy_with_logits(model.logits(x, k=k), x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def plot_filters(model, path):
    """Saves the learned filters as an 11x11 grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    w = model.encoder.weight.detach().cpu()
    w = (w - w.amin(dim=(1, 2, 3), keepdim=True)) / (
        w.amax(dim=(1, 2, 3), keepdim=True) - w.amin(dim=(1, 2, 3), keepdim=True) + 1e-8)
    grid = make_grid(w, nrow=11, padding=1, pad_value=1.0)[0]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def main(argv=None):
    args = parse_args(argv)
    device = utils.resolve_device(args.device)
    utils.set_seed(args.seed)

    dataset = data.background_set(args.data_dir, centre=not args.no_centre)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers,
                        generator=torch.Generator().manual_seed(args.seed))

    model = modules.ECPretrain().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    print("Device: {}. {} batches of {}.".format(device, args.batches, args.batch_size))
    start, done = time.time(), 0
    loss_avg = utils.RunningAverage()
    while done < args.batches:
        # The paper counts batches, not epochs, so loop the loader as needed.
        for batch in loader:
            loss_avg.update(pretrain_step(model, optimizer, batch, device, args.k))
            done += 1
            if done % 100 == 0:
                print("batch {:>5}  loss {:.4f}  {:.0f} s".format(
                    done, loss_avg(), time.time() - start))
                loss_avg.reset()
            if done == args.batches:
                break

    # Only the model weights go in the file. Optimizer state would triple
    # the size and nothing downstream needs it.
    state = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
             "args": vars(args)}
    path = utils.save_checkpoint(state, args.out_dir, args.name, silent=False)
    plot_filters(model, args.filters_plot)
    print("Wrote {} and {}".format(path, args.filters_plot))


if __name__ == "__main__":
    main()
