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
import models.camera
import models.medium_model
import models.morphable_model
from torchvision.transforms import Grayscale
from torch.nn import functional as F

import random
import sys
sys.path.append('./models')
import models
from torchvision.io import read_image

import multiframe_reconstructror



from utils import utils

rdir = '/online_data/face/yt_faces2/yt_cropped2'#sys.argv[2]
sdir = '/online_data/face/yt_faces2/yt_cropped2'#sys.argv[2]
# sdir = '/online_data/face/yt_faces/yt_cropped/'#sys.argv[2]
dbname = os.path.basename(rdir)

checkpoint_dir = 'checkpoints_norm'
device = 'cuda'

fov = 20
cx = 112
cy = 112

# backbone = 'resnet18'
backbone = 'resnet50'
which_bfm = 'BFMmm-23660'
# checkpoint = torch.load('checkpoints_norm/medium_model30.00yt_cropped2.pth')
# init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{which_bfm}.pth'
init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{which_bfm}_02381.pth'
init_path_id = f'{checkpoint_dir}/medium_model20.00combined_celeb_ytfacesresnet5081866True1e-05BFMmm-23660.pth'

# init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{which_bfm}_00671.pth'

checkpoint = torch.load(init_path_id)
# checkpoint = torch.load('checkpoints_norm/medium_modelsv_0046430.00allyt_cropped.pth',  map_location='cpu')
model = models.medium_model.MediumModel(rasterize_fov=fov, rasterize_size=2*cx, 
                                        label_means=checkpoint['label_means'].to(device), 
                                        label_stds=checkpoint['label_stds'].to(device),
                                        which_bfm=which_bfm,
                                        which_backbone=backbone)


mm = models.morphable_model.MorphableModel(device=device, key=which_bfm)
# mm.to(device)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

transform_gray = Grayscale(num_output_channels=3)

camera = models.camera.Camera(fov, fov, cx, cy)


#%%
plt.figure(figsize=(30*0.8,10*0.8))
sdir = '/offline_data/face/synth/identity/fov020/'#sys.argv[2]
# tdir = '/offline_data/face/synth/identity/output/fov020/3DIDv2-M01'
tdir = '/offline_data/face/synth/identity/output/fov020/3DIDv5-M01'



import scipy

if not os.path.exists(tdir):
    os.mkdir(tdir)

imfs = glob(f'{sdir}/id003*jpg')
imfs.sort()
off = 300

alpha_list = []
us_list = []
taus_list = []

N = len(imfs)
# random.shuffle(imfs)
# for fi, imf in enumerate([imfs[10], imfs[500], imfs[1200]]):
# for fi, imf in enumerate(imfs[int(N*0.75)::10]):
# for fi, imf in enumerate(imfs[::50]):
    
target_ps = []
for fi, imf in enumerate(imfs[:7]):
    print(f'\r{fi}')
    bn = '.'.join(os.path.basename(imf).split('.')[:-1])
    tpath = f'{tdir}/{bn}.mat'
    tpath_txt = f'{tdir}/{bn}.txt'
    
    # if os.path.exists(tpath) and os.path.exists(tpath_txt):
        # continue
    
    im = cv2.imread(imf)
    cim = im.astype(np.float32)/255.0
    lf = imf.replace('.jpg', '.txt')
    lmks = np.loadtxt(lf)
    
    if len(lmks.shape) == 1:
        xs = lmks[::2]
        ys = lmks[1::2]
    else:
        xs = lmks[:,0]
        ys = lmks[:,1]
    
    ys = cim.shape[0]-ys
    lmks68 = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)
    M = utils.estimate_norm(lmks68[17:,:], cim.shape[0], 1.5, [25,25])
    cim = utils.resize_n_crop_cv(cim, M, 224)
    cim = np.transpose(cim, (2,0,1))
    cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
    cim = cim.to(device)
    y = model(cim)
    
    params = model.parse_params(y[0])
    
    
    
    # p0 = mm.compute_face_shape(0*params['alpha'], 0*params['exp'])
    # p0 = p0.detach().cpu().squeeze().numpy()
    
    p = mm.compute_face_shape(params['alpha'], params['exp'])
    
    mask, _, rim  = model.render_image(params)[0]
    p = p.detach().cpu().squeeze().numpy()
    # plt.clf()
    plt.figure(fi, figsize=(30*1.5,10*1.5))

    cim0 = cim.cpu().numpy()[0,0,:,:]    
    rim0 = rim.detach().cpu().numpy()[0,0,:,:]
    mask = mask.detach().cpu().numpy()[0,0,:,:]
    
    
    # plt.plot(params['exp'][0].detach().cpu(), alpha=0.4)
    plt.subplot(161)
    plt.imshow(cim0)
    
    plt.subplot(162)
    plt.imshow(rim0)
    
    rim0[mask==1] = (cim0[mask==1]+2*rim0[mask==1])/3.0
    
    us_list.append(params['angles'].detach())
    taus_list.append(params['tau'].detach())

    cps = model.mm.project_to_2d(camera, params['angles'], params['tau'], params['alpha'])
    target_ps.append(cps)
    
    alpha_list.append(params['alpha'])
    
    x = torch.cat([params['angles'], params['tau']], dim=1)
    # pts = reconstructor(x).detach().cpu().squeeze().numpy()
    
    plt.subplot(163)
    plt.imshow(rim0)
    # plt.plot(pts[:,0], 224-pts[:,1], '.')
    
    
    plt.subplot(164)
    # plt.plot(pts[:,0], pts[:,1], '.')



    plt.subplot(165)
    # plt.plot(p0[:,0], p0[:,1], '.')
    plt.plot(p[:,0], p[:,1], '.')
    plt.ylim((-90, 90))
    
    
    plt.subplot(166)
    # plt.plot(p0[:,2], p0[:,1], '.')
    plt.plot(p[:,2], p[:,1], '.')
    plt.ylim((-90, 90))
    

    plt.show()

#%%

alpha = torch.cat(alpha_list).mean(dim=0).unsqueeze(0)
us = torch.cat(us_list, 1)
taus = torch.cat(taus_list, 1)

fov2 = fov/1.0

cam1 = models.camera.Camera(fov, fov, cx, cy)
cam2 = models.camera.Camera(fov2, fov2, cx, cy)

taus[:,2::3] *= cam2.f_x/cam1.f_x
#%%
reconstructor = multiframe_reconstructror.MultiframeReconstructor(fov2, fov2, cx, cy, us, taus, alpha.detach())
reconstructor.to(device)
    
#%%

reconstructor.alpha.requires_grad = False

target = reconstructor.gather_targets(target_ps).detach()
optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-1)

losses = []
for n in range(500):
    output = reconstructor()
    loss = F.l1_loss(target,output)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if n % 100 == 0:
        print(f'{n}: {loss.item():.4f}')
    
        
    #     print(f'{n}: {loss.item():.4f}')
    #     p = reconstructor.compute_face_shape().detach().cpu().squeeze()
    #     plt.figure(figsize=(40, 20))
    #     plt.subplot(121)
    #     plt.plot(p[:,0], p[:,1], '.')
    #     plt.xlim((-100, 100))
    #     plt.ylim((-100, 100))
    #     plt.subplot(122)
    #     plt.plot(p[:,2], p[:,1], '.')
    #     plt.xlim((-100, 100))
    #     plt.ylim((-100, 100))
    #     plt.show()


reconstructor.alpha.requires_grad = True
#%%


target = reconstructor.gather_targets(target_ps).detach()
optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-1)
# optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-2)

losses = []
for n in range(2000):
    output = reconstructor()
    loss = F.l1_loss(target,output)+0.1*torch.mean((1./(reconstructor.mm.sigma_alphas))*torch.abs(reconstructor.alpha))
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print(reconstructor.alpha)
    if n % 500 == 0:
        print(f'{n}: {loss.item():.4f} - {torch.max(reconstructor.alpha)}')

        p = reconstructor.compute_face_shape().detach().cpu().squeeze()
        plt.figure(figsize=(40, 40))
        plt.subplot(221)
        plt.plot(p[:,0], p[:,1], '.')
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))
        plt.subplot(222)
        plt.plot(p[:,2], p[:,1], '.')
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))
        # plt.show()
        plt.subplot(212)
        plt.stem(reconstructor.alpha.detach().cpu().squeeze(), '.')
        # plt.xlim((-100, 100))
        # plt.ylim((-100, 100))
        plt.show()

    """           """
    
    # m = {'pts': p}
    # scipy.io.savemat(tpath, m)
    # np.savetxt(tpath_txt, p)
#%%

target = reconstructor.gather_targets(target_ps).detach()
# optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-2)
optimizer = torch.optim.LBFGS(reconstructor.parameters(), lr=1e-1)


def f_obj(output, alpha):
    return F.l1_loss(target,output)+0.1*torch.mean((1./(reconstructor.mm.sigma_alphas))*torch.abs(reconstructor.alpha))
    
# L-BFGS
def closure():
    optimizer.zero_grad()
    objective = f_obj(reconstructor(), reconstructor.alpha)
    objective.backward()
    return objective




losses = []
for n in range(20000):
    # output = reconstructor()
    # loss = F.l1_loss(target,output)+0.1*torch.mean((1./(reconstructor.mm.sigma_alphas))*torch.abs(reconstructor.alpha))
    # losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(closure)
    
    # print(reconstructor.alpha)
    if n % 10 == 0:
        print(f'{n}: {loss.item():.4f} - {torch.max(reconstructor.alpha)}')

        p = reconstructor.compute_face_shape().detach().cpu().squeeze()
        plt.figure(figsize=(40, 40))
        plt.subplot(221)
        plt.plot(p[:,0], p[:,1], '.')
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))
        plt.subplot(222)
        plt.plot(p[:,2], p[:,1], '.')
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))
        # plt.show()
        plt.subplot(212)
        plt.stem(reconstructor.alpha.detach().cpu().squeeze(), '.')
        # plt.xlim((-100, 100))
        # plt.ylim((-100, 100))
        plt.show()

