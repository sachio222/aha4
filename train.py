"""
Title: train.py
Author: J. Krajewski
Copyright: 2020
License:
Description: Training for model implemented from 2019 paper
             "AHA an Artificial Hippocampal Algorithm for Episodic Machine 
             Learning" by Kowaldo, Ahmed and Rawlinson.

             Runs image through EC, DG / EC-->CA3, CA3, and CA1. This network
             accepts multiple, varied examples of an original sample, and
             reconstructs the original sample, given multiple samples that vary
             from the origin sample to different degrees.

             Pretraining was run on EC prior to running this model (see
             pretrain.py)
             
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
# print("\033c", end="") # (Doesn't fully clear screen on PC)
print("\033[H\033[2J", end="")

# Initialize paths to json parameters
data_path = Path().absolute() / "data"
model_path = Path().absolute() / "experiments/train/"
pretrain_path = Path().absolute() / "experiments/pretrain/"
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


def train(model,
          dataloader,
          ectoca3_optimizer,
          ca1_optimizer,
          ectoca3_loss_fn,
          ca1_loss_fn,
          params,
          autosave=False,
          train_mode=True):

    # Set model to train or eval.
    if not train_mode:
        model.eval()

        # Load weights
        utils.load_checkpoint(model_path, step1_ec, name="ectoca3_weights")
        utils.load_checkpoint(model_path, step5_ca1, name="ca1_weights")

        # Custom loader for ca3.
        ca3_weights_path = model_path / "ca3_weights.pth.tar"
        ca3_weights = torch.load(ca3_weights_path.as_posix())
        step3_ca3.W = ca3_weights
    else:
        model.train()

    for epoch in range(params.num_epochs):
        for i, (x, _) in enumerate(dataloader):
            if params.cuda:
                x = x.cuda(non_blocking=True)

            #=============RUN EC=============#

            with torch.no_grad():
                ec_maxpool_flat = step1_ec(x, k=4)

            #=====MONITORING=====#

            # ec_out_weight = step1_ec.encoder.weight.data
            ## DISPLAY
            # utils.animate_weights(ec_out_weight, auto=False)

            # for i, out in enumerate(ec_maxpool_flat):
            #     print(out.shape)
            #     ec_grid = torchvision.utils.make_grid(out, nrow=11)
            #     utils.animate_weights(ec_grid, i)

            #=====END MONIT.=====#

            #=============END EC=============#

            #=============RUN DENTATE GYRUS=============#
            with torch.no_grad():
                dg_sparse = step2_dg(ec_maxpool_flat, k=10)

            ## DISPLAY 
            # utils.showme(dg_sparse)
            # exit()

            # Polarize output from (0, 1) to (-1, 1) for step3_ca3
            dg_sparse_dressed = modules.all_dressed(dg_sparse)

            ## DISPLAY 
            # utils.showme(dg_sparse_dressed)
            # exit()

            #=============END DENTATE GYRUS=============#

            #=============RUN CA3 TRAINING==============#

            if not train_mode:
                pass
            else:
                with torch.no_grad():
                    ca3_weights = step3_ca3.train(dg_sparse_dressed)

                if autosave:
                    ca3_state = step3_ca3.W
                    utils.save_checkpoint(ca3_state,
                                          model_path,
                                          name="ca3_weights",
                                          silent=False)
            
            ## DISPLAY
            # utils.showme(ca3_weights)
            # exit()

            #=============END CA3 TRAINING==============#

            #=============RUN EC->CA3===================#

            if not train_mode:
                trained_sparse = step4_ectoca3(ec_maxpool_flat)
            else:
                # Run training
                for i in range(params.ectoca3_iters):
                    trained_sparse = step4_ectoca3(ec_maxpool_flat)
                    ectoca3_loss = ectoca3_loss_fn(trained_sparse, dg_sparse)
                    ectoca3_optimizer.zero_grad()
                    ectoca3_loss.backward(retain_graph=True)
                    print(i, ectoca3_loss)
                    # NOTE: Learning rate has large impact on quality of output
                    ectoca3_optimizer.step()


                    utils.animate_weights(trained_sparse.detach())

                if autosave:
                    ec_state = utils.get_save_state(epoch, step4_ectoca3,
                                                    ectoca3_optimizer)
                    utils.save_checkpoint(ec_state,
                                          model_path,
                                          name="ectoca3_weights",
                                          silent=False)

            # Polarize output from (0, 1) to (-1, 1) for step3_ca3
            ectoca3_out_dressed = modules.all_dressed(trained_sparse)

            ## DISPLAY
            # utils.showme(ectoca3_out_dressed.detach())
            # exit()

            #=============END EC->CA3=================#

            #=============RUN CA3 RECALL==============#

            ca3_out_recall = step3_ca3.update(ectoca3_out_dressed)

            ## DISPLAY
            # utils.showme(ca3_out_recall.detach())

            #=============END CA3 TRAINING==============#

            #=============RUN CA1 ======================#

            if not train_mode:
                ca1_reconstruction = step5_ca1(ca3_out_recall)
            else:
                for i in range(params.ca1_iters):
                    ca1_reconstruction = step5_ca1(ca3_out_recall)
                    ca1_loss = ca1_loss_fn(ca1_reconstruction, x)
                    ca1_optimizer.zero_grad()

                    if i == (params.ca1_iters - 1):
                        ca1_loss.backward(retain_graph=False)
                        print("Graph cleared.\n")
                    else:
                        ca1_loss.backward(retain_graph=True)

                    print(i, ca1_loss)
                    ca1_optimizer.step()

                    ## DISPLAY
                    utils.animate_weights(ca1_reconstruction.detach(), nrow=5)

                if autosave:
                    ec_state = utils.get_save_state(epoch, step5_ca1,
                                                    ectoca3_optimizer)
                    utils.save_checkpoint(ec_state,
                                          model_path,
                                          name="ca1_weights",
                                          silent=False)

                #=============END CA1 =============#

            # Optional exit to end after one batch
            exit()


# Define transforms
tsfm = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(params.resize_dim),
    transforms.ToTensor()
])

# Import from torchvision.datasets Omniglot
dataset = Omniglot(data_path, background=False, transform=tsfm, download=True)

dataloader = DataLoader(dataset,
                        params.batch_size,
                        shuffle=True,
                        num_workers=params.num_workers,
                        drop_last=True)

#================BEGIN MODELS================#

# Initialize layers with parmeters.
step1_ec = modules.EC(params.batch_size,
                      D_in=1,
                      D_out=121,
                      KERNEL_SIZE=9,
                      STRIDE=1,
                      PADDING=1)

step2_dg = modules.DG(params.batch_size, 27225, 225)

step3_ca3 = modules.CA3(225)

step4_ectoca3 = modules.ECToCA3(27225, 225)

step5_ca1 = modules.CA1(params.batch_size, 225, 2704, params.resize_dim)

#================END MODELS================#

# Set loss_fn to Binary cross entropy for Autoencoder.
ectoca3_loss_fn = nn.BCELoss()
ca1_loss_fn = nn.MSELoss()

ectoca3_optimizer = optim.Adam(step4_ectoca3.parameters(),
                               lr=params.ectoca3_learning_rate,
                               weight_decay=params.ectoca3_weight_decay)

ca1_optimizer = optim.Adam(step5_ca1.parameters(),
                           lr=params.ca1_learning_rate,
                           weight_decay=params.ca1_weight_decay)

# Get pretrained weights. Comment out if not wanted.
utils.load_checkpoint(pretrain_path, step1_ec, name="pre_train")

# Start training
# Train mode runs backprop and stores weights in the Hopfield net. 
# Autosave over-writes existing weights if set to true.

train(step1_ec,
      dataloader,
      ectoca3_optimizer,
      ca1_optimizer,
      ectoca3_loss_fn,
      ca1_loss_fn,
      params,
      autosave=False,
      train_mode=True)
