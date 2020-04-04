"""
Title: pre-train.py
Author: J. Krajewski
Copyright: 2020
License:
Description: Pretraining for model implemented from 2019 paper
             "AHA an Artificial Hippocampal Algorithm for Episodic Machine 
             Learning" by Kowadlo, Ahmed and Rawlinson.

             Trains visual cortex on Omniglot dataset.

Thanks: Special thanks to Gideon Kowaldo and David Rawlinson for
             putting me up to the task of recreating their model in Pytorch!
             As well as @ptrblk in the Pytorch forums for always coming through
             with answers when I needed them. 
"""

from pathlib2 import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import Omniglot

import matplotlib.pyplot as plt
from tqdm import tqdm

# User modules
from model import modules  # pylint: disable=no-name-in-module
from utils import utils  # pylint: disable=RP0003, F0401

# Clear terminal with ANSI <ESC> c "\033c"
# print("\033c", end="") # Doesn't work on PC
print("\033[H\033[2J", end="")

# Initialize paths to json parameters
data_path = Path().absolute() / "data"
model_path = Path().absolute() / "experiments/pretrain/"
json_path = model_path / "params.json"

# Load params json
assert json_path.is_file(
), f"\n\nERROR: No params.json file found at {json_path}\n"
params = utils.Params(json_path)

# If GPU, write to params file
params.cuda = torch.cuda.is_available()

# Set random seed
torch.manual_seed(42)
if params.cuda:
    torch.cuda.manual_seed(42)
    # Update num_workers to 2 if running on GPU
    params.num_workers = 2


def train(model, dataloader, optimizer, loss_fn, params, autosave=True):
    # Set model to train or eval.
    model.train()

    for epoch in range(params.num_epochs):

        loss_avg = utils.RunningAverage()
        desc = "Epoch: {}".format(epoch)  # Informational only, used in tqdm.

        with tqdm(desc=desc, total=len(dataloader)) as t:
            for i, (x, _) in enumerate(dataloader):
                if params. cuda:
                    x, _ = x.cuda(non_blocking=True)

                # GK note that currently k is ignored y model
                y_pred = model(x, k=4)    # GK does this call forward() ?

                # Set loss comparison to input x
                loss = loss_fn(y_pred, x)

                optimizer.zero_grad()
                loss.backward()
 
                #=====MONITORING=====#

                enc_weights = model.encoder.weight.data
                # utils.animate_weights(enc_weights, label=i, auto=False)
                # for s in range(len(x)):
                #     utils.animate_weights(y_pred[s].detach(), label=i, auto=True)
                
                #=====END MONIT.=====#

                optimizer.step()
                loss_avg.update(loss.item())

                # Update tqdm progress bar.
                t.set_postfix(loss="{:05.8f}".format(loss_avg()))
                t.update()

            # Show one last time
            # utils.animate_weights(enc_weights, auto=False)

        if autosave:
            # Autosaves latest state after each epoch (overwrites previous state)
            state = utils.get_save_state(epoch, model, optimizer)
            utils.save_checkpoint(state,
                                  model_path,
                                  name=f"pre_train_{params.batch_size}",
                                  silent=False)

        # grid_img = torchvision.utils.make_grid(y_pred, nrow=8)
        # plt.imshow(grid_img.detach().numpy()[0])
        # plt.show()

# Define transforms
tsfm = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(params.resize_dim),
    transforms.ToTensor()
])

# Import from torchvision.datasets Omniglot
dataset = Omniglot(data_path, background=True, transform=tsfm, download=True)
dataloader = DataLoader(dataset,
                        params.batch_size,
                        shuffle=True,
                        num_workers=params.num_workers,
                        drop_last=True)

# Load visual cortex model here.
model = modules.ECPretrain(D_in=1,
                           D_out=121,
                           KERNEL_SIZE=9,
                           STRIDE=5,
                           PADDING=1)

# GK:  BCELoss may work better, but FYI we used mse (more tested for topk sparse mode), which is probably better suited to unbounded values (i.e. linear rather than sigmoid activation function)

# Set loss_fn to Binary cross entropy for Autoencoder.
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

# Get last trained weights. COMMENT OUT if not wanted
# utils.load_checkpoint(model_path, model, optimizer, name=f"pre_train_{params.batch_size}")

# Start training
train(model, dataloader, optimizer, loss_fn, params, autosave=True)



