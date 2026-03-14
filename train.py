#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 2024
@author: Aobo Li, Hope Fu

This script trains deep learning model over the training dataset, possible architecture includes:
 - Fully Connected Network [fcnet]
 - Positional U-Net [punet]
 - Transformer [transformer]
"""

import numpy as np
import argparse
from datetime import date
import torch
import torch.nn as nn
import h5py
# import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
import math
from torch.utils.data import dataset
from network import PositionalUNet, FocalLoss1D, TransformerModel, AE, SimpleWaveNet, RNNSeq2Seq, S4DenoisModel, MixtureMSESpectralLoss
import wandb

# SQUID = h5py.File(SQUIDname,'r')
# SG = h5py.File(SGname, 'r')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")

# Output directory with default as current directory
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Denoising model we would like to train [fcnet/punet/transformer] (Default: punet).')
parser.add_argument('-w', '--weak', action='store_true', help='Train model on the weak version of the datasets.')
parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging.')
parser.add_argument('--wandb_project', type=str, default='TIDMAD', help='W&B project name (default: TIDMAD).')

args = parser.parse_args()

#set the size of segmentations for deep learning models
input_size = 40000
if args.denoising_model == "transformer":
    input_size = 20000 # transformer model requires additional GPU memories, so we reduce segment size by 50%

if args.wandb:
    wandb.init(project=args.wandb_project, config={
        "model": args.denoising_model,
        "input_size": input_size,
        "weak": args.weak,
    }, name=f"{args.denoising_model}_{date.today()}")
sample_size = 10 #Randomly sample 20% of the time series to train model
batchsize = 1
output_size = input_size
ADC_CHANNEL = 256

def normalize(time_series):
    time_series = time_series[::100] #subsample a shorter TS to calculate mean and std
    return time_series.mean(), time_series.std()

def read_loader(ABRAfile):
    alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries'])+128
    alltarget = np.array(ABRAfile['timeseries']['channel0002']['timeseries'])+128

    max_index = 2000000000
    alltrain = alltrain[:max_index].reshape( -1,sample_size, batchsize, input_size)
    alltarget = alltarget[:max_index].reshape(-1,sample_size, batchsize, input_size)
    random_index = np.random.randint(sample_size)

    return np.concatenate([alltrain[:,random_index],alltarget[:,random_index]],axis=1)

'''
Due to the large frequency variation, we have to train different models for different frequency ranges.
Files with higher file numbers have higher frequencies, so we train 4 models per architecture to adapt to
different frequency ranges:
 - Low Frequency: {Model}_0_4.pth
 - Medium Frequency: {Model}_4_10.pth
 - Medium-High Frequency: {Model}_10_15.pth
 - High Frequency: {Model}_15_20.pth
For more detail, please read appendix A of the paper
'''
ifile_checkpoint = [0,4,10,15,20]
# ifile_checkpoint = [0, 20]

global_step = 0
file_list = []
rb, re = (0,20)
if args.weak:
    rb += 20
    re += 19
    ifile_checkpoint = [20,24,30,35,39]
for ifile in range(rb,re):
    if ifile<10:
        fname = f"abra_training_000{ifile}.h5"
    elif ifile<100:
        fname = f"abra_training_00{ifile}.h5"
    else:
        fname = f"abra_training_0{ifile}.h5"
    if not os.path.exists(os.path.join(args.data_dir,fname)):
        continue

    if ifile in ifile_checkpoint:
        # Initialize a new model at the beginning of new checkpoint
        if args.denoising_model == "punet":
            model = PositionalUNet().to(DEVICE)
            criterion = FocalLoss1D().to(DEVICE)
        elif args.denoising_model == "transformer":
            model = TransformerModel().to(DEVICE)
            criterion = FocalLoss1D().to(DEVICE)
        elif args.denoising_model == "fcnet":
            model = AE(input_size).to(DEVICE)
            criterion = nn.SmoothL1Loss().to(DEVICE)
        elif args.denoising_model == "wavenet":
            model = SimpleWaveNet().to(DEVICE)
            criterion = FocalLoss1D().to(DEVICE)
        elif args.denoising_model == "rnn":
            model = RNNSeq2Seq().to(DEVICE)
            criterion = FocalLoss1D().to(DEVICE)
        elif args.denoising_model == "s4denois":
            model = S4DenoisModel().to(DEVICE)
            criterion = MixtureMSESpectralLoss().to(DEVICE)
        else:
            raise ValueError

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


    # Read file (retry on transient network/IO errors)
    fpath = os.path.join(args.data_dir, fname)
    all_data = None
    for attempt in range(5):
        try:
            ABRAfile = h5py.File(fpath, 'r')
            all_data = read_loader(ABRAfile)
            ABRAfile.close()
            break
        except OSError as e:
            wait = 2 ** attempt
            print(f'WARNING: {fname} read attempt {attempt+1}/5 failed ({e}), retrying in {wait}s...')
            import time; time.sleep(wait)
    if all_data is None:
        print(f'ERROR: Skipping {fname} after 5 failed attempts.')
        continue
    np.random.shuffle(all_data)
    val_split = max(1, int(0.1 * len(all_data)))
    val_data = all_data[-val_split:]
    train_data = all_data[:-val_split]

    model.train()
    for i, batch in enumerate(train_data):
        inputarr, targetarr = (batch[:batchsize], batch[batchsize:])
        input_seq = torch.from_numpy(inputarr)
        target_seq = torch.from_numpy(targetarr)
        randind = np.random.randint(batchsize)
        input_seq = input_seq[randind].unsqueeze(0).float().to(DEVICE)
        target_seq = target_seq[randind].unsqueeze(0).float().to(DEVICE)

        # Forward pass
        if not (args.denoising_model in ["fcnet", "s4denois"]):
            input_seq = input_seq.int()
            target_seq = target_seq.long()

        output_seq = model(input_seq)
        # Calculate the loss
        loss = criterion(output_seq, target_seq)

        # Backward pass and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # Print and log the loss every 500 batches
        if i % 500 == 50:
            print('Epoch: {} | Batch: {} | Loss: {}'.format(ifile, i, loss.item()))
            if args.wandb:
                wandb.log({"train/loss": loss.item(), "file": ifile}, step=global_step)

    # Compute validation loss at the end of each file
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_data:
            inputarr, targetarr = (batch[:batchsize], batch[batchsize:])
            input_seq = torch.from_numpy(inputarr)
            target_seq = torch.from_numpy(targetarr)
            randind = np.random.randint(batchsize)
            input_seq = input_seq[randind].unsqueeze(0).float().to(DEVICE)
            target_seq = target_seq[randind].unsqueeze(0).float().to(DEVICE)
            if not (args.denoising_model in ["fcnet", "s4denois"]):
                input_seq = input_seq.int()
                target_seq = target_seq.long()
            output_seq = model(input_seq)
            val_losses.append(criterion(output_seq, target_seq).item())
    mean_val_loss = np.mean(val_losses)
    print('Epoch: {} | Val Loss: {}'.format(ifile, mean_val_loss))
    if args.wandb:
        wandb.log({"val/loss": mean_val_loss, "file": ifile}, step=global_step)

    del ABRAfile, all_data, train_data, val_data
    gc.collect()

    # Save and delete current model at the end of current checkpoint
    if (ifile+1) in ifile_checkpoint:
        prev_file = ifile_checkpoint[ifile_checkpoint.index(ifile+1)-1]
        if args.denoising_model == "punet":
            torch.save(model, f'PUNet_{prev_file}_{ifile+1}.pth')
        elif args.denoising_model == "fcnet":
            torch.save(model, f'FCNet_{prev_file}_{ifile+1}.pth')
        elif args.denoising_model == "transformer":
            torch.save(model, f'Transformer_{prev_file}_{ifile+1}.pth')
        elif args.denoising_model == "wavenet":
            torch.save(model, f'WaveNet_{prev_file}_{ifile+1}.pth')
        elif args.denoising_model == "rnn":
            torch.save(model, f'RNN_{prev_file}_{ifile+1}.pth')
        elif args.denoising_model == "s4denois":
            torch.save(model, f'S4Denois_{prev_file}_{ifile+1}.pth')
        del model, criterion, optimizer
        torch.cuda.empty_cache()