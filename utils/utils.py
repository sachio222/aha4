"""Small helpers for devices, seeds and checkpoints.

Checkpoint helpers started from the Stanford CS230 project guidelines:
https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
"""

import random
from pathlib import Path

import numpy as np
import torch


class RunningAverage():
    """A simple class that maintains the running average of a qty.

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

    reset = __init__


def resolve_device(name="auto"):
    """Turns "auto", "cpu", "mps" or "cuda" into a torch.device.

    "auto" picks CUDA if there is an NVIDIA GPU, then MPS (Apple Silicon),
    then CPU. Asking for a backend this machine doesn't have raises a
    RuntimeError up front instead of failing halfway through a run.
    """
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available here.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available here.")
    return torch.device(name)


def to_device(batch, device):
    """Moves the images of a batch to device.

    Dataloaders hand back (images, labels). The old code did
    `x, _ = x.cuda()`, which unpacks the image tensor itself along the batch.
    """
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    return images.to(device, non_blocking=True)


def set_seed(seed):
    """Seeds Python, NumPy and PyTorch so runs repeat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_save_state(epoch, model, optimizer):
    """Bundles model and optimizer state for save_checkpoint.

    Raises ValueError if the optimizer doesn't hold the model's parameters,
    which is how the CA1 checkpoint ended up with the PR optimizer.
    """
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not all(id(p) in owned for p in trainable):
        raise ValueError("This optimizer does not own the model's parameters.")

    return {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict()
    }


def save_checkpoint(state, checkpoint, name="last", silent=True):
    """Saves a state dict to checkpoint/name.pth.tar.

    Args:
        state: (dict) contains the state dict, and optionally epoch and optimizer dict
        checkpoint: (Path) directory to store saved states, created if missing
        name: (string) prefix to '.pth.tar' eg: name.pth.tar
        silent: (bool) if True, skip output messages
    """
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint / "{}.pth.tar".format(name)
    torch.save(state, filepath)
    if not silent:
        print("Saved {}".format(filepath))
    return filepath


def load_checkpoint(checkpoint, model, optimizer=None, name="last", map_location="cpu"):
    """Loads parameters from checkpoint/name.pth.tar into model (and optimizer).

    Args:
        checkpoint: (Path) directory holding the file
        model: (torch.nn.Module) model to load parameters into
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        name: (string) prefix to '.pth.tar' eg: name.pth.tar
        map_location: where to put the tensors, "cpu" works everywhere
    """
    filepath = Path(checkpoint) / "{}.pth.tar".format(name)
    if not filepath.is_file():
        raise FileNotFoundError("No checkpoint at {}".format(filepath))

    state = torch.load(filepath, map_location=map_location)
    model.load_state_dict(state["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(state["optim_dict"])

    return state
