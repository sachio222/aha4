# utils.py
# Jacob Krajewski, 2020
#
# Large portions inspired from Stanford CS2300 best practices guidelines
# for deep learning projects. Original repository available below:
# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

import json
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib2 import Path


class Params():

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by 'params.dict['learning_rate]."""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a qty.

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
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


def show_sample_img(dataset, idx):
    sample = dataset.__getitem__(idx)
    plt.imshow(sample[0].numpy()[0])
    plt.show()


def print_full_tensor(tensor):
    """You know how it only shows part of the tensor when you print?

    Well use this to show the whole thing.
    """

    torch.set_printoptions(profile="full")
    print(tensor)
    torch.set_printoptions(profile="default")


def get_save_state(epoch, model, optimizer):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict()
    }
    return state


def save_checkpoint(state, checkpoint, name="last", silent=True):
    """Saves state dict to file. 

    Args:
        state: (dict) contains epoch, state dict and optimizer dict
        checkpoint: (Path) directory name to store saved states
        name: (string) previx to '.pth.tar' eg: name.pth.tar
        silent: (bool) if True, bypass output messages

    Todo:
        Simplify the silent checks so I don't need 4 if statements
    """
    filepath = checkpoint / "{}.pth.tar".format(name)
    if not Path(checkpoint).exists():
        if not silent:
            print("Creating checkpoint directory {}".format(checkpoint))
        Path(checkpoint).mkdir()
    else:
        if not silent:
            print("Getting checkpoint directory...")
    if not silent:
        print("Saving file...")
    # Remember to convert filepath to str or it flips out when trying to save
    torch.save(state, str(filepath))
    if not silent:
        print("File saved successfully.")


def load_checkpoint(checkpoint, model, optimizer=None, name="last"):
    """Loads parameters dict from checkpoint file to model, and optimizer.

    Args:
        checkpoint: (string) filename to load
        model: (torch.nn.Module) model to load parameters
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        name: (string) previx to '.pth.tar' eg: name.pth.tar
    """
    filepath = checkpoint / "{}.pth.tar".format(name)

    print("Looking for saved files...", end=" ")

    if not Path(checkpoint).exists():
        raise ("File does not exist at {}".format(checkpoint))
    checkpoint = torch.load(str(filepath))

    print("Found.")

    model.load_state_dict(checkpoint.get("state_dict"), strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint.get("optim_dict"))

    print("Loading saved weights complete.")
    return checkpoint


def showme(tnsr,
           size_dim0=10,
           size_dim1=10,
           title=None,
           full=False,
           detach=False,
           grid=False):
    """Does all the nasty matplotlib stuff for free. 
    """
    if detach:
        tnsr = tnsr.detach().numpy()

    if not grid:
        if len(tnsr.shape) > 2:
            tnsr = tnsr.view(tnsr.shape[0], -1)

        fig, ax = plt.subplots(figsize=(size_dim0, size_dim1))
        ax.set_title(title, color="blue", loc="left", pad=20)
        ax.matshow(tnsr)
        plt.show()
        print(tnsr.shape)
        if full:
            print(tnsr)
    else:
        grid_img = torchvision.utils.make_grid(tnsr, nrow=5)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        print(tnsr.shape)


def animate_weights(t, nrow=11, label=None, auto=False):
    """Animates weights during training. Only works on Mac.

    Press ctrl + C in terminal to escape. Change auto to True if you are 
    running on a mac. It is pretty good. 

        Usage example:
            >>> animate_weights(conv1_weights, i)

    Args:
        t: (tensor) from "model.layer_name.weight.data" on layer
        iter: (scalar, string) Optional. Shows label for each pass
    """

    grid_img = torchvision.utils.make_grid(t, nrow)
    # plt.ion()
    plt.title(label, color="blue", loc="left", pad=20)
    plt.imshow(grid_img.numpy()[0])
    if not auto:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(0.0001)
        plt.close()


def accuracy(study, test):
    """ compute the accuracy mse matching between study and train set"""
    import numpy as np

    correct_count = 0
    for i, sample in enumerate(study):

        # TODO: I'm not sure to detach, or to make into np !!! but this seemed to work .... for now
        error = sample - test
        sq_error = np.square(error.detach())
        mse = torch.mean(sq_error, dim=1)
        min_idx = np.argmin(mse)

        print("i={}, min_idx={}".format(i, min_idx))
        # This is a special case when the matching test sample is at the same index as the study sample
        if min_idx == i:
            correct_count += 1

    acc = correct_count / len(study)
    return acc


def plot_it(vals):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, len(vals))
    fig, ax = plt.subplots()
    train_loss, = ax.plot(x, vals, '--', linewidth=2, label='ec-ca3 loss')
    ax.legend(loc='upper right')
    plt.show()
