#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:51:13 2023

@author: v
"""
import os
import sys
import copy
import torch
import numpy as np

from torchvision.transforms import Grayscale
from torch.nn import functional as F
from torch.utils.data import DataLoader
from time import time
from data import DirectDataset

import matplotlib.pyplot as plt
from models import morphable_model, separated_model

fov = 20

cx = 112
rasterize_size = 2*cx
rdir =  '/online_data/face/yt_faces2/yt_cropped2'
# rdir =  '/online_data/face/yt_faces/yt_cropped'
rdir = '/online_data/face/combined_celeb_ytfaces'

dbname = os.path.basename(rdir)
normalize_labels = True
device = 'cuda'

which_bfm='BFMmm-23660'
which_model = 'sep_modelv3'
backbone = 'resnet50'

learning_rate = 1e-5

dbname = os.path.basename(rdir)
which_bfm = 'BFMmm-23660'
which_resnet = 'resnet50'
normalize_labels = True
device = 'cuda'
tform_data = True


train_data = DirectDataset(fov, rasterize_size, transform=Grayscale(num_output_channels=3), is_train=True, 
                           normalize_labels=normalize_labels, rootdir=rdir, which_bfm=which_bfm, do_tform=tform_data)
test_data = DirectDataset(fov, rasterize_size, transform=Grayscale(num_output_channels=3), is_train=False, 
                          normalize_labels=normalize_labels, rootdir=rdir, which_bfm=which_bfm)


if not normalize_labels:
    checkpoint_dir = 'checkpoints/'
else:
    checkpoint_dir = 'checkpoints_norm/'
    
    
label_stds = None
label_means = None

if normalize_labels:
    label_stds = torch.from_numpy(train_data.stds).to(device)
    label_means = torch.from_numpy(train_data.means).to(device)

Ntra = 81866
learning_rate = 1e-5


# init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{which_bfm}_02381.pth'
init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{Ntra}{tform_data}{learning_rate}{which_bfm}.pth'

init_path_perframe = init_path_id

# which_model = 'sep_modelv0'
# which_model = 'sep_modelv1'


if which_model == 'sep_modelv0':
    model = separated_model.SeparatedModelV0(rasterize_fov=fov, rasterize_size=rasterize_size,
                                           init_path_id=init_path_id, init_path_perframe=init_path_perframe,
                                           label_stds=label_stds, label_means=label_means,
                                           which_bfm=which_bfm, backbone=backbone)

elif which_model == 'sep_modelv1':
    model = separated_model.SeparatedModel(rasterize_fov=fov, rasterize_size=rasterize_size,
                                           init_path_id=init_path_id, init_path_perframe=init_path_perframe,
                                           label_stds=label_stds, label_means=label_means,
                                           which_bfm=which_bfm, backbone=backbone)

elif which_model == 'sep_modelv3':
    model = separated_model.SeparatedModelV3(rasterize_fov=fov, rasterize_size=rasterize_size,
                                           init_path_id=init_path_id, init_path_perframe=init_path_perframe,
                                           label_stds=label_stds, label_means=label_means,
                                           which_bfm=which_bfm, which_backbone=backbone)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

model = model.to(device)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.scheduler_step_size)



# model.freeze_all_but_rigid_layers()
hist_tes = {'rigid':[], 'all':[], 'nonrigid': []}
hist_tra = {'rigid':[], 'all':[], 'nonrigid': [] }

# finetune_rigid = True
phase = 'all'
# phase = 'rigid'
# phase = 'nonrigid'


os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_file0 = f'{checkpoint_dir}/{which_model}SP{fov:.2f}{dbname}{backbone}{learning_rate}{tform_data}{Ntra}.pth'

if os.path.exists(checkpoint_file0):
    checkpoint = torch.load(checkpoint_file0)
    model.load_state_dict(checkpoint['model_state'])
    hist_tes = checkpoint['hist_tes']
    hist_tra = checkpoint['hist_tra']
    
    print(f'Loaded checkpoint file {checkpoint_file0}')



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%



for n in range(0, 10000):
    print(f'{n} ({len(hist_tes[phase])}) {which_model}')
    
    train_loss = 0
    model.train()
    num_batches = 0
    t0 = time()
    for train_features, train_labels in train_dataloader:
        inputs  = train_features.to(device)
        targets = train_labels.to(device)
        
        output, alpha, beta, inputs2 = model(inputs, tforms=None, render=False)
        closs = F.l1_loss(output[:,:79], targets[:,199:(199+79)]) + F.l1_loss(output[:,-6:], targets[:,-6:])

        train_loss += closs.item()

        optimizer.zero_grad()
        closs.backward()
        optimizer.step()
        num_batches += 1
        if num_batches == 800:
            break
        
    # plt.figure()
    # plt.imshow(rim[0,0,:,:].detach().cpu())
    # plt.figure()
    # plt.imshow(rim[0,1,:,:].detach().cpu())
    # plt.figure()
    # plt.imshow(rim[0,2,:,:].detach().cpu())
    
    
    hist_tra[phase].append(train_loss/num_batches)
    
    model.eval()
    
    with torch.no_grad():
        
        tes_loss = 0
        num_batches = 0
        for test_features, test_labels in test_dataloader:
            inputs  = test_features.to(device)
            targets = test_labels.to(device)
            
            output, alpha, beta, inputs2 = model(inputs, None, render=False)
            params = model.parse_params(output, alpha, beta)
            closs = F.l1_loss(output[:,:79], targets[:,199:(199+79)]) + F.l1_loss(output[:,-6:], targets[:,-6:])
                            
            tes_loss += closs.item() #* inputs.size(0)
            num_batches += 1
            
        
        hist_tes[phase].append(tes_loss/num_batches)
        
        if n > 2 and (hist_tes[phase][-1] < min(hist_tes[phase][-n:-1])):
            checkpoint_file = f'{checkpoint_dir}/{which_model}vSP_{n:05d}{fov:.2f}{phase}{dbname}.pth'
            
            checkpoint = {
                "model_state": model.state_dict(),
                "hist_tra": hist_tra,
                "hist_tes": hist_tes,
                "nb_epochs_finished": n,
                "cuda_rng_state": torch.cuda.get_rng_state(),
                "label_means": label_means,
                "label_stds": label_stds
            }
            torch.save(checkpoint, checkpoint_file0)
            # torch.save(checkpoint, checkpoint_file)
            
            print(f'saved {checkpoint_file0}')
        
        
        output_id = model.backbone_id(inputs)[0]
        params_est_id = model.backbone_id.parse_params(output_id)
        
        
        params_est = model.parse_params(output, alpha, beta)
        params_gt = model.backbone_id.parse_params(targets)
        
        (_, _, ims_gt), pr = model.backbone_id.render_image(params_gt)
        (_, _, ims_est), pr = model.backbone_id.render_image(params_est)
        (_, _, ims_est_id), pr = model.backbone_id.render_image(params_est_id)
        
        params_est0 = copy.deepcopy(params_est)
        params_est0['exp'] *= 0
        (_, _, ims_est0), pr = model.backbone_id.render_image(params_est0)
        
        plt.figure(figsize=(50,10))
        plt.subplot(151)
        plt.imshow(ims_gt[0].detach().cpu().numpy()[0,:,:])
        plt.subplot(152)
        plt.imshow(ims_est_id[0].detach().cpu().numpy()[0,:,:])
        plt.subplot(153)
        plt.imshow(ims_est0[0].detach().cpu().numpy()[0,:,:])
        plt.subplot(154)
        plt.imshow(ims_est[0].detach().cpu().numpy()[0,:,:])
        plt.subplot(155)
        plt.imshow(inputs[0].detach().cpu().numpy()[0,:,:])
        plt.show()
    
        plt.clf()
        plt.subplot(121)
        plt.semilogy(hist_tra[phase][-n:])
        plt.subplot(122)
        plt.semilogy(hist_tes[phase][-n:])
            
        plt.show()
    
    print('%.2f seconds' % (time()-t0))

#%%

for param in model.rigid_layers.parameters():
    print(param.weights)



