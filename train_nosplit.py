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
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from network import PositionalUNet, FocalLoss1D, TransformerModel, AE, SimpleWaveNet, RNNSeq2Seq, S4DenoisModel, MixtureMSESpectralLoss
import itertools
import wandb
# import psutil
# import os

# # Get current process
# process = psutil.Process(os.getpid())
# print(f"Memory percentage: {process.memory_percent():.2f}%")
# SQUID = h5py.File(SQUIDname,'r')
# SG = h5py.File(SGname, 'r')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")

# Output directory with default as current directory
parser.add_argument('--data_dir', '-d', type=str, default="/home/klz/Data/TIDMAD/", help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Denoising model we would like to train [fcnet/punet/transformer] (Default: punet).')
parser.add_argument('-f', '--force', action='store_true', help='Directly proceed to download without asking the confirming question.')
parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging.')
parser.add_argument('--wandb_project', type=str, default='TIDMAD', help='W&B project name (default: TIDMAD).')

args = parser.parse_args()

#set the size of segmentations for deep learning models
input_size = 40000
if args.denoising_model == "transformer":
	input_size = 20000 # transformer model requires additional GPU memories, so we reduce segment size by 50%
sample_size = 5 #Randomly sample (100/20)% of the time series to train model
batchsize = 10
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

class TIDMADDataset(Dataset):

    def __init__(self, fpath, fname, idict, tdict):
        '''
        input: directory to signal file and directory to background file

        '''
        self.filepath = fpath
        self.filelist = fname
        self.seglength = batchsize * input_size
        self.idict = {}
        self.tdict = {}

        self.train = self.pull_event_from_dir(self.filelist)
        np.random.shuffle(self.train)
        self.size = len(self.train)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        event_tuple = self.train[idx]
        input_series = self.idict[event_tuple[0]][int(event_tuple[1])]
        target_series = self.tdict[event_tuple[0]][int(event_tuple[1])]
        #     input_series = np.array(f['timeseries']['channel0001']['timeseries'])[start:end].reshape(batchsize, input_size)
        #     target_series = np.array(f['timeseries']['channel0002']['timeseries'])[start:end].reshape(batchsize, input_size)

        return input_series.astype(np.int16)+128, target_series.astype(np.int16)+128

    def return_time_channel(self):
        return (self.__getitem__(0)[0].shape[0], self.__getitem__(0)[0].shape[1])

    def pull_event_from_dir(self,filelist):

        # alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries'])+128
        # alltarget = np.array(ABRAfile['timeseries']['channel0002']['timeseries'])+128

        # max_index = 2000000000
        # alltrain = alltrain[:max_index].reshape( -1,sample_size, batchsize, input_size)
        # alltarget = alltarget[:max_index].reshape(-1,sample_size, batchsize, input_size)
        max_index = 2000000000
        indices_array = np.arange(max_index).astype(int).reshape( -1,sample_size, batchsize, input_size)
        evlist = []
        for filename in tqdm(filelist):
            # print(self.fpath, filename)
            with h5py.File(os.path.join(self.filepath, filename), 'r') as ABRAfile:
                alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries']).astype(np.int8)
                alltarget = np.array(ABRAfile['timeseries']['channel0002']['timeseries']).astype(np.int8)
                max_index = 2000000000
                random_index = np.random.randint(sample_size)
                self.idict[filename] = alltrain[:max_index].reshape( -1,sample_size, batchsize, input_size)[:,random_index].copy()
                self.tdict[filename] = alltarget[:max_index].reshape(-1,sample_size, batchsize, input_size)[:,random_index].copy()
                iarray = np.arange(self.idict[filename].shape[0])
                evlist += list(zip(itertools.repeat(filename), iarray))
                del alltrain, alltarget
                gc.collect()
            gc.collect()
            # print(f"Memory percentage: {process.memory_percent():.2f}%")
        return evlist

file_list = []
input_map = {}
target_map = {}
for ifile in range(0,20):
    if ifile<10:
        fname = f"abra_training_000{ifile}.h5"
    elif ifile<100:
        fname = f"abra_training_00{ifile}.h5"
    else:
        fname = f"abra_training_0{ifile}.h5"
    if not os.path.exists(os.path.join(args.data_dir,fname)):
        continue
    file_list.append(fname)

dataset = TIDMADDataset(args.data_dir, file_list, input_map, target_map)
dataset_size = len(dataset)
indices = list(range(dataset_size))

np.random.shuffle(indices)
val_split = max(1, int(0.1 * dataset_size))
train_indices, val_indices = indices[val_split:], indices[:val_split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler, drop_last=True)

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

if args.wandb:
    wandb.init(project=args.wandb_project, config={
        "model": args.denoising_model,
        "input_size": input_size,
    }, name=f"{args.denoising_model}_{date.today()}")

model.train()
for i, batch in tqdm(enumerate(train_loader)):
    inputarr, targetarr = (batch[0][0], batch[1][0])
    input_seq = inputarr.float().to(DEVICE)
    target_seq = targetarr.float().to(DEVICE)

    # Forward pass
    if not args.denoising_model in ["fcnet", "s4denois"]:
        input_seq = input_seq.int()
        target_seq = target_seq.long()

    output_seq = model(input_seq)
    # Calculate the loss
    loss = criterion(output_seq, target_seq)

    # Backward pass and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print and log the loss every 50 batches
    if i % 50 == 0:
        print('Batch: {} | Loss: {}'.format(i, loss.item()))
        if args.wandb:
            wandb.log({"train/loss": loss.item()}, step=i)

    # Compute and log validation loss every 500 batches
    if i % 500 == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_input, val_target = (val_batch[0][0].float().to(DEVICE), val_batch[1][0].float().to(DEVICE))
                if not args.denoising_model in ["fcnet", "s4denois"]:
                    val_input = val_input.int()
                    val_target = val_target.long()
                val_output = model(val_input)
                val_losses.append(criterion(val_output, val_target).item())
        mean_val_loss = np.mean(val_losses)
        print('Batch: {} | Val Loss: {}'.format(i, mean_val_loss))
        if args.wandb:
            wandb.log({"val/loss": mean_val_loss}, step=i)
        model.train()

del train_loader, val_loader
gc.collect()

if args.denoising_model == "punet":
    torch.save(model, f'PUNet_0_20.pth')
elif args.denoising_model == "fcnet":
    torch.save(model, f'FCNet_0_20.pth')
elif args.denoising_model == "transformer":
    torch.save(model, f'Transformer_0_20.pth')
elif args.denoising_model == "wavenet":
    torch.save(model, f'WaveNet_0_20.pth')
elif args.denoising_model == "rnn":
    torch.save(model, f'RNN_0_20.pth')
elif args.denoising_model == "s4denois":
    torch.save(model, f'S4Denois_0_20.pth')
del model, criterion, optimizer
torch.cuda.empty_cache()
