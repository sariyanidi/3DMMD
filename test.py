#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:37:06 2023

@author: sariyanide
"""

import numpy as np
import torch
from glob import glob
import matplotlib.pyplot as plt
import cv2
import os
from math import floor, ceil
import models.medium_model
import models.morphable_model
from torchvision.transforms import Grayscale


import random
import sys
sys.path.append('./models')
import models
from torchvision.io import read_image


#%%

from utils import utils

sdir = '/online_data/face/yt_faces2/yt_cropped2/'#sys.argv[2]
sdir = '/online_data/face/yt_faces/yt_cropped/'#sys.argv[2]

device = 'cuda'

# checkpoint = torch.load('checkpoints_norm/medium_model30.00yt_cropped2.pth')
checkpoint = torch.load('checkpoints_norm/medium_model30.00yt_cropped.pth')
# checkpoint = torch.load('checkpoints_norm/medium_modelsv_0046430.00allyt_cropped.pth',  map_location='cpu')
model = models.medium_model.MediumModel(rasterize_fov=30, rasterize_size=224, 
                                        label_means=checkpoint['label_means'].to(device), 
                                        label_stds=checkpoint['label_stds'].to(device))


mm = models.morphable_model.MorphableModel(device=device)
# mm.to(device)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

transform_gray = Grayscale(num_output_channels=3)

#%%
plt.figure(figsize=(30*0.8,10*0.8))


imfs = glob(f'{sdir}/*jpg')
imfs.sort()
off = 300

N = len(imfs)
# random.shuffle(imfs)
# for fi, imf in enumerate([imfs[10], imfs[500], imfs[1200]]):
for fi, imf in enumerate(imfs[int(N*0.75)::40]):

    bimf = os.path.basename(imf)

    # if os.path.exists(dstim_path):
        # continue

    im = cv2.imread(imf)
    cim = im.astype(np.float32)/255.0
    lf = imf.replace('.jpg', '.txt')
    lmks = np.loadtxt(lf)
    xs = lmks[::2]
    ys = im.shape[0]-lmks[1::2]
    lmks68 = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)
    M = utils.estimate_norm(lmks68[17:,:], im.shape[0], 1.5, [25,25])
    # cim = utils.resize_n_crop_cv(im, M, 224).astype(np.float32)/255.0
    cim = np.transpose(cim, (2,0,1))
    cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
    cim = cim.to(device)
    y = model(cim)
    params = model.parse_params(y[0])
    
    # plt.clf()
    plt.figure(fi, figsize=(30*1.5,10*1.5))
    
    p0 = mm.compute_face_shape(0*params['alpha'], 0*params['exp'])
    p0 = p0.detach().cpu().squeeze().numpy()
    
    p = mm.compute_face_shape(params['alpha'], params['exp'])
    p = p.detach().cpu().squeeze().numpy()
    # plt.plot(params['exp'][0].detach().cpu(), alpha=0.4)
    plt.subplot(131)
    plt.imshow(cim.cpu().numpy()[0,0,:,:])
    
    plt.subplot(132)
    # plt.plot(p0[:,0], p0[:,1], '.')
    plt.plot(p[:,0], p[:,1], '.')
    plt.ylim((-90, 90))
    
    
    plt.subplot(133)
    # plt.plot(p0[:,2], p0[:,1], '.')
    plt.plot(p[:,2], p[:,1], '.')
    plt.ylim((-90, 90))
    """   """
    
    plt.show()
    # break