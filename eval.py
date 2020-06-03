"""
Title: train.py
Author: J. Krajewski
Copyright: 2020
License:
Description: Training for model implemented from 2019 paper
             "AHA an Artificial Hippocampal Algorithm for Episodic Machine 
             Learning" by Kowadlo, Ahmed and Rawlinson.

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

ideal_vals_array = []

# Initialize paths to json parameters
data_path = Path().absolute() / "data"
model_path = Path().absolute() / "experiments/train/"
pretrain_path = Path().absolute() / "experiments/pretrain/"
json_path = model_path / "params.json"
json_path_pretrain = pretrain_path / "params.json"

# Load params json
assert json_path.is_file(
), f"\n\nERROR: No params.json file found at {json_path}\n"
params = utils.Params(json_path)
pretrain_params = utils.Params(json_path_pretrain)

# If GPU, write to params file
params.cuda = torch.cuda.is_available()

# Set random seed
torch.manual_seed(42)
if params.cuda:
    torch.cuda.manual_seed(42)
    # Update num_workers to 2 if running on GPU
    params.num_workers = 2


def train(dataloader,
          ectoca3_optimizer,
          ca1_optimizer,
          ectoca3_loss_fn,
          ca1_loss_fn,
          params,
          autosave=False,
          train_mode=True,
          display=False):

    # Set model to train or eval.
    if not train_mode:
        print("Setting to eval mode.")
        step1_ec.eval()
        step4_ectoca3.eval()
        step5_ca1.eval()

        # Load weights
        utils.load_checkpoint(model_path, step4_ectoca3, name="ectoca3_weights")
        utils.load_checkpoint(model_path, step5_ca1, name="ca1_weights")

        # Custom loader for ca3.
        ca3_weights_path = model_path / "ca3_weights.pth.tar"
        ca3_weights = torch.load(ca3_weights_path.as_posix())
        step3_ca3.W = ca3_weights

    else:
        step1_ec.train()          # GK: can i confirm that this is not actually training, just doing a forward pass?
        step4_ectoca3.train()
        step5_ca1.train()

    for epoch in range(params.num_epochs):
        for i, x in enumerate(dataloader):

            if i >= params.batches:
                break

            ideal_vals = {}
            if params.cuda:
                x = x.cuda(non_blocking=True)

            #=============RUN EC=============#
            # entire dataloader is train set for eval. 
            x = dataloader

            if display:
                pass
                utils.animate_weights(x, nrow=5)

            with torch.no_grad():
                ec_maxpool_flat = step1_ec(x, k=4)
                ideal_vals['ec'] = ec_maxpool_flat

            if display:
                utils.animate_weights(step1_ec.encoder.weight.data, nrow=11)
                # exit()
     
                # for i, out in enumerate(ec_maxpool_flat):
                #     ec_grid = torchvision.utils.make_grid(out, nrow=11)
                #     utils.animate_weights(ec_grid, i, auto=True)
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
            if display:
                utils.showme(dg_sparse, title="DG OUT")
                # exit()

            # Polarize output from (0, 1) to (-1, 1) for step3_ca3
            dg_sparse_dressed = modules.all_dressed(dg_sparse)

            ## DISPLAY 
            if display:
                utils.showme(dg_sparse_dressed, title="DG CLEAN")
                # exit()

            #=============END DENTATE GYRUS=============#



            #=============RUN CA3 TRAINING==============#

            # GK: this needs to do multiple training iterations with the same DG input (as does EC-CA3)
            # GK: is that the case here?

            if not train_mode:
                pass
            else:
                with torch.no_grad():
                    ca3_weights = step3_ca3.train(dg_sparse_dressed, "pinverse")

                if autosave:
                    ca3_state = step3_ca3.W
                    utils.save_checkpoint(ca3_state,
                                          model_path,
                                          name="ca3_weights",
                                          silent=False)
                
                print("CA3 weights updated.")
            
            ## DISPLAY
            if display:
                utils.showme(ca3_weights, title="Weights")
                # exit()

            ideal_vals['ca3'] = dg_sparse_dressed

            #=============END CA3 TRAINING==============#



            #=============RUN EC->CA3===================#

            if not train_mode:
                trained_sparse = step4_ectoca3(ec_maxpool_flat)
                trained_sparse = modules.get_top_k(trained_sparse, k=10, topk_dim=1, scatter_dim=1)

                # torch.set_printoptions(profile="full")
                # print(f"dg_sparse: {dg_sparse[3]}")
                # print(f"trained: {trained_sparse[3]}")
                # print(f"trained: {trained_sparse[3].max()}")
                # torch.set_printoptions(profile="default")

                ## DISPLAY
                if display:
                    utils.showme(trained_sparse.detach(), title="Trained Prediction")
                    # exit()
            else:
                # Run training
                loss_avg = utils.RunningAverage()
                ectoca3_loss_history = []

                with tqdm(desc="Updating EC->CA3", total=params.ectoca3_iters) as t1:
                    trained_sparse = None
                    for i in range(params.ectoca3_iters):
                        trained_sparse = step4_ectoca3(ec_maxpool_flat)
                        ectoca3_loss = ectoca3_loss_fn(trained_sparse, dg_sparse)
                        ectoca3_optimizer.zero_grad()
                        ectoca3_loss.backward(retain_graph=True)
                        # print(i, ectoca3_loss)
                        # NOTE: Learning rate has large impact on quality of output
                        ectoca3_optimizer.step()

                        loss_avg.update(ectoca3_loss.item())

                        t1.set_postfix(loss="{:05.3f}".format(loss_avg()))
                        t1.update()

                        ## DISPLAY
                        if display:
                            utils.animate_weights(trained_sparse.detach(), auto=False)

                        ectoca3_loss_history.append(ectoca3_loss)

                    ideal_vals['ectoca3'] = trained_sparse

                if autosave:
                    ec_state = utils.get_save_state(epoch, step4_ectoca3,
                                                    ectoca3_optimizer)
                    utils.save_checkpoint(ec_state,
                                          model_path,
                                          name="ectoca3_weights",
                                          silent=False)

                if display:
                    utils.plot_it(ectoca3_loss_history)
                # TODO: run a forward pass on test set, get loss, and plot on the same plot (learning curves)


            # Polarize output from (0, 1) to (-1, 1) for step3_ca3
            # ectoca3_out_dressed = modules.center_me_zero(trained_sparse)
            ectoca3_out_dressed = modules.all_dressed(trained_sparse)

            ## DISPLAY
            if display:
                utils.showme(ectoca3_out_dressed.detach(), title="Cleaned-Trained")
                # exit()

            #=============END EC->CA3=================#



            #=============RUN CA3 RECALL==============#

            ca3_out_recall = step3_ca3.update(ectoca3_out_dressed)
            # ca3_out_recall = step3_ca3.update(dg_sparse_dressed)

            ## DISPLAY
            if display:
                utils.showme(ca3_out_recall.detach(), title="Hopfield out")
                # exit()

            #=============END CA3 TRAINING==============#



            #=============RUN CA1 ======================#

            if not train_mode:
                ca1_reconstruction = step5_ca1(ca3_out_recall)
                utils.animate_weights(ca1_reconstruction.detach(), nrow=5, auto=False)
                exit()
            else:
                loss_avg.reset()

                with tqdm (desc="Updating CA1", total=params.ca1_iters) as t2:
                    ca1_reconstruction = None
                    for i in range(params.ca1_iters):
                        ca1_reconstruction = step5_ca1(ca3_out_recall)
                        ca1_loss = ca1_loss_fn(ca1_reconstruction, x)
                        ca1_optimizer.zero_grad()

                        if i == (params.ca1_iters - 1):
                            ca1_loss.backward(retain_graph=False)
                        else:
                            ca1_loss.backward(retain_graph=True)

                        # print(i, ca1_loss)
                        ca1_optimizer.step()

                        loss_avg.update(ca1_loss.item())

                        t2.set_postfix(loss="{:05.3f}".format(loss_avg()))
                        t2.update()

                        ## DISPLAY
                        if display:
                            utils.animate_weights(ca1_reconstruction.detach(), nrow=5, auto=False)

                    ideal_vals['ca1'] = ca1_reconstruction

                if autosave:
                    ec_state = utils.get_save_state(epoch, step5_ca1,
                                                    ectoca3_optimizer)
                    utils.save_checkpoint(ec_state,
                                          model_path,
                                          name="ca1_weights",
                                          silent=False)

                print("Graph cleared.", end=" ")
                print("Weights successfully updated.\n")

            ## DISPLAY
            if display:
                utils.animate_weights(ca1_reconstruction.detach(), nrow=5, auto=False)

                #=============END CA1 =============#

            ideal_vals_array.append(ideal_vals)

            # Optional exit to end after one batch
            # exit()

    import pickle
    file = open(r'ideal_vals.pkl', 'wb')
    pickle.dump(ideal_vals_array, file)
    file.close()




# Define transforms
tsfm = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(params.resize_dim),
    transforms.ToTensor()
])

# Import from torchvision.datasets Omniglot
dataset = Omniglot(data_path, background=False, transform=tsfm, download=True)

test_dataset = []

torch.manual_seed(params.test_seed)
# Get batch_size random samples
idxs = torch.randint(len(dataset), (1, params.batch_size))
# Make sure one of them is our control.
idxs[0, 0] = 0


for i, idx in enumerate(idxs[0]):
    test_dataset.append(dataset[idx + params.train_shift][0][0])
    # utils.animate_weights(test_dataset[i], auto=True)

dataloader = torch.stack(test_dataset)
dataloader.unsqueeze_(1)
# print(dataloader.shape)

#================BEGIN MODELS================#

# Initialize layers with parameters.
step1_ec = modules.EC(params.batch_size,
                      D_in=1,
                      D_out=121,
                      KERNEL_SIZE=9,
                      STRIDE=1,
                      PADDING=4)

step2_dg = modules.DG(params.batch_size, 20449, 225)

step3_ca3 = modules.CA3(225)

step4_ectoca3 = modules.ECToCA3(20449, 225)

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
utils.load_checkpoint(pretrain_path, step1_ec, name=f"pre_train_{pretrain_params.batch_size}")

# Start training
# Train mode runs backprop and stores weights in the Hopfield net.
# Autosave over-writes existing weights if set to true.

train(dataloader,
      ectoca3_optimizer,
      ca1_optimizer,
      ectoca3_loss_fn,
      ca1_loss_fn,
      params,
      autosave=True,
      train_mode=True,
      display=False)
