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

rdir = '/online_data/face/yt_faces2/yt_cropped2' # sys.argv[2]
sdir = '/online_data/face/yt_faces2/yt_cropped2' # sys.argv[2]

# sdir = '/online_data/face/yt_faces/yt_cropped/' # sys.argv[2]

dbname = os.path.basename(rdir)

checkpoint_dir = 'checkpoints_norm'
device = 'cuda'

fov = 15
cx = 112
cy = 112



# backbone = 'resnet18'
backbone = 'resnet50'
which_bfm = 'BFMmm-23660'
# checkpoint = torch.load('checkpoints_norm/medium_model30.00yt_cropped2.pth')
# init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{which_bfm}.pth'
init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{which_bfm}_02381.pth'
init_path_id = f'{checkpoint_dir}/medium_model20.00combined_celeb_ytfacesresnet5081866True1e-05BFMmm-23660.pth'
init_path_id = f'{checkpoint_dir}/medium_model15.00combined_celeb_ytfacesresnet50351639True1e-05-2-BFMmm-23660UNL.pth'
init_path_id = f'{checkpoint_dir}/medium_model15.00combined_celeb_ytfacesresnet50-2-BFMmm-23660_01377.pth'
init_path_id = f'{checkpoint_dir}/medium_model15.00combined_celeb_ytfacesresnet50139979True1e-05-2-BFMmm-23660UNL.pth'
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
rasterize_size = 2*camera.cx


#%%




plt.figure(figsize=(30*0.8,10*0.8))

sdir = '/offline_data/face/synth/identity/fov020/'#sys.argv[2]
# tdir = '/offline_data/face/synth/identity/output/fov020/3DIDv2-M01'
tdir = '/offline_data/face/synth/identity/output/fov020/3DIDv5-M01'

all_shapes_persubj = {}
all_pts_persubj = {}
all_invMs_persubj = {}
all_ims_persubj = {}

import scipy

if not os.path.exists(tdir):
    os.mkdir(tdir)

# 33, 38, 41
# subj_id = 3
for subj_id in range(0,100):
    all_shapes_persubj[subj_id] = []
    all_pts_persubj[subj_id] = []
    all_invMs_persubj[subj_id] = []
    all_ims_persubj[subj_id] = []
    imfs = glob(f'{sdir}/id{subj_id:03d}*jpg')
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
    for fi, imf in enumerate(imfs):
        print(imf)
        print(f'\r{fi} - {subj_id}')
        bn = '.'.join(os.path.basename(imf).split('.')[:-1])
        tpath = f'{tdir}/{bn}.mat'
        tpath_txt = f'{tdir}/{bn}.txt'
        
        if not os.path.exists('tmp3'):
            os.mkdir('tmp3')
        tmp_path = f'tmp3/subj{subj_id:03d}_{fi}.txt'
        
        # if os.path.exists(tmp_path):
            # continue
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
        
        #ys = cim.shape[0]-ys
        lmks68 = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)
        
        M = utils.estimate_norm(lmks68[17:,:], cim.shape[0], 1.5, [25,25])
        cim = utils.resize_n_crop_cv(cim, M, int(rasterize_size))
        invM = utils.estimate_inv_norm(lmks68[17:,:], im.shape[1], 1.5, [25,25])
        
        cim = np.transpose(cim, (2,0,1))
        cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
        cim = cim.to(device)
        y = model(cim)
        
        params = model.parse_params(y[0])
        
        # p0 = mm.compute_face_shape(0*params['alpha'], 0*params['exp'])
        # p0 = p0.detach().cpu().squeeze().numpy()
        
        p = mm.compute_face_shape(params['alpha'], params['exp'])
        
        
        euler_angles = mm.mrp_to_euler(params['angles'].detach().cpu())
        if np.abs(euler_angles[2]) > 28 or np.abs(euler_angles[1]) > 18:
            continue
        
        mask, _, rim  = model.render_image(params)[0]
        p = p.detach().cpu().squeeze().numpy()
        # plt.clf()
        # plt.figure(fi, figsize=(50*1.5,10*1.5))
    
        cim0 = cim.cpu().numpy()[0,0,:,:]    
        rim0 = rim.detach().cpu().numpy()[0,0,:,:]
        mask = mask.detach().cpu().numpy()[0,0,:,:]
        
        
        """
        # plt.plot(params['exp'][0].detach().cpu(), alpha=0.4)
        plt.subplot(161)
        plt.imshow(cim0)
        
        plt.subplot(162)
        plt.imshow(rim0)
        """
        
        rim0[mask==1] = (cim0[mask==1]+2*rim0[mask==1])/3.0
        
        us_list.append(params['angles'].detach())
        taus_list.append(params['tau'].detach())
    
        cps = model.mm.project_to_2d(camera, params['angles'], params['tau'], params['alpha'], params['exp'])
        all_shapes_persubj[subj_id].append(p)
        all_pts_persubj[subj_id].append(cps.detach().cpu())
        all_invMs_persubj[subj_id].append(invM)
        all_ims_persubj[subj_id].append(im)
        """
        target_ps.append(cps)
        
        alpha_list.append(params['alpha'])
        
        x = torch.cat([params['angles'], params['tau']], dim=1)
        # pts = reconstructor(x).detach().cpu().squeeze().numpy()
                
        cps_ = cps.detach().cpu().squeeze().numpy()
        np.savetxt(tmp_path, cps_)
        np.savetxt(f'tmp/R{subj_id:03d}_{fi}.txt', p)
        
        plt.subplot(163)
        plt.imshow(rim0)
        # plt.plot(pts[:,0], 224-pts[:,1], '.')
        
        
        
        plt.subplot(164)
        plt.plot(cps_[:,0], cps_[:,1], '.')
    
        plt.subplot(165)
        # plt.plot(p0[:,0], p0[:,1], '.')
        plt.plot(p[:,0], p[:,1], '.')
        plt.ylim((-90, 90))
        
        plt.subplot(166)
        # plt.plot(p0[:,2], p0[:,1], '.')
        plt.plot(p[:,2], p[:,1], '.')
        plt.ylim((-90, 90))
        
        plt.show()
        # break
    # break
"""










#%%

device = 'cuda'



outdir_final = 'tmppp'
if not os.path.exists(outdir_final):
    os.mkdir(outdir_final)

import models.orthograhic_fitter
import models.perspective_fitter


def compute_neutral_face(mm, all_pts_input, all_invMs_input, all_ims_input, subj_id):
    
    which_pts = 'sampled'
    Nframes = 22
    fov = 20
    tdir = f'/offline_data/face/synth/identity/output/fov020/3DIDv1-fov{fov:02d}-{which_pts}-M{Nframes:02d}'
    os.makedirs(tdir, exist_ok=True)
    tpath_txt = f'{tdir}/id{subj_id:03d}_001.txt'
    
    if os.path.exists(tpath_txt):
        return
    
    
    cam_rec = models.camera.Camera(fov_x=fov, fov_y=fov, cx=320, cy=320)
    
    alpha_path = f'{outdir_final}/{subj_id}.alpha'
    
    if os.path.exists(alpha_path):
        return np.loadtxt(alpha_path)
    
    random.seed(1907)
    of = models.orthograhic_fitter.OrthographicFitter(mm)
    
    pfs_f = models.perspective_fitter.PerspectiveFitter(mm, cam_rec, use_maha=True, which_pts='lmks', F=1)
    pfd_f = models.perspective_fitter.PerspectiveFitter(mm, cam_rec, use_maha=True, which_pts=which_pts, F=1)
    
    alphas = []
    taus = []
    us = []
    
    all_pts = []
    
    print(len(all_pts_persubj[subj_id]))
    for ptsi in range(min(len(all_pts_input), Nframes)):
        
        pts = all_pts_input[ptsi].clone().squeeze().numpy()
        invM = all_invMs_input[ptsi]
        
        pts[:,1] = rasterize_size-pts[:,1]
        pts = np.concatenate((pts, torch.ones((pts.shape[0], 1))), axis=1)
        
        pts = (invM @ pts.T).T
        pts[:,1] = all_ims_input[ptsi].shape[1]-pts[:,1]
        """
        plt.figure(figsize=(50,50))
        plt.imshow(all_ims_input[ptsi])
        plt.plot(pts[:,0], pts[:,1], '.')
        """
        
        lmks = torch.from_numpy(pts).to(device)[mm.li,:]
        # lmks[:,1] = 1920-lmks[:,1]
        of_fit_params = of.fit_orthographic_GN(lmks.cpu().numpy(), plotit=False)[0]
        u = torch.tensor(of_fit_params['u'])
        tau = of.to_projective(of_fit_params, cam_rec.get_matrix(), lmks.cpu().float())
    
        x0 = torch.cat([0*torch.rand(pfs_f.num_components), torch.tensor(tau), torch.tensor(u)]).to(device)
        # x0 = pfd_f.fit_GN(x0.float(), [torch.from_numpy(pts).to(mm.device).float()], plotit=True, use_ineq = False)[0]
        # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = True)[0]
        # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = False)[0]
        # break
        # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = True)[0]
        x0, fit_out = pfd_f.fit_GN(x0.float(), [torch.from_numpy(pts).to(device).float()], 
                                   plotit=False, plotlast=False, use_ineq = True)
        # break
        
        alphas.append(x0[:pfs_f.num_components])
        taus.append(x0[pfs_f.num_components:pfs_f.num_components+3])
        us.append(x0[pfs_f.num_components+3:pfs_f.num_components+6])
        
        all_pts.append(torch.from_numpy(pts).to(device).float())
        # break
    # return
    
    idx = np.arange(len(taus)).astype(int)
    random.shuffle(idx)
    idx = idx[:Nframes]
    calphas = [alphas[x] for x in idx]
    ctaus = [taus[x] for x in idx]
    cus = [us[x] for x in idx]
    call_pts = [all_pts[x] for x in idx]
    
    pfd = models.perspective_fitter.PerspectiveFitter(mm, cam_rec, use_maha=True, 
                                                      which_pts=which_pts, 
                                                      F=Nframes)
    x0 = pfd.construct_initialization(calphas, ctaus, cus)
    x, fit_out = pfd.fit_GN(x0.float(), call_pts, plotit=False, plotlast=False, use_ineq = True)
    
    alpha = x[:pfd.num_components].reshape(1,-1)
    
    
    p0 = mm.compute_face_shape(alpha).squeeze().cpu().numpy()
    
    # m = {'pts': p0}
    # scipy.io.savemat(tpath, m)
    np.savetxt(tpath_txt, p0)

    




def compute_neutral_face_multi(mm, all_pts_input, all_invMs_input, all_ims_input, subj_id):
    
    which_pts = 'lmks'
    Nframes = 12
    fov = 20
    tdir = f'/offline_data/face/synth/identity/output/fov020/3DIDv2-fov{fov:02d}-{which_pts}-M{Nframes:02d}'
    os.makedirs(tdir, exist_ok=True)
    tpath_txt = f'{tdir}/id{subj_id:03d}_001.txt'
    
    if os.path.exists(tpath_txt):
        return
    
    
    cam_rec = models.camera.Camera(fov_x=fov, fov_y=fov, cx=320, cy=320)
    
    alpha_path = f'{outdir_final}/{subj_id}.alpha'
    
    if os.path.exists(alpha_path):
        return np.loadtxt(alpha_path)
    
    random.seed(1907)
    of = models.orthograhic_fitter.OrthographicFitter(mm)
    
    pfs_f = models.perspective_fitter.PerspectiveFitter(mm, cam_rec, use_maha=True, which_pts='lmks', F=1)
    pfd_f = models.perspective_fitter.PerspectiveFitter(mm, cam_rec, use_maha=True, which_pts=which_pts, F=1)
    
    alphas = []
    taus = []
    us = []
    
    all_pts = []
    Nframes_  = min(len(all_pts_input), Nframes)
    
    cands = np.arange(Nframes_)
    groups = [random.sample(cands.tolist(), 12) for y in range(7)]
    
    p0s = []
    for group in groups:
        print(group)
        for ptsi in group:
            
            pts = all_pts_input[ptsi].clone().squeeze().numpy()
            invM = all_invMs_input[ptsi]
            
            pts[:,1] = rasterize_size-pts[:,1]
            pts = np.concatenate((pts, torch.ones((pts.shape[0], 1))), axis=1)
            
            pts = (invM @ pts.T).T
            pts[:,1] = all_ims_input[ptsi].shape[1]-pts[:,1]
            """
            plt.figure(figsize=(50,50))
            plt.imshow(all_ims_input[ptsi])
            plt.plot(pts[:,0], pts[:,1], '.')
            """
            
            lmks = torch.from_numpy(pts).to(device)[mm.li,:]
            of_fit_params = of.fit_orthographic_GN(lmks.cpu().numpy(), plotit=False)[0]
            u = torch.tensor(of_fit_params['u'])
            tau = of.to_projective(of_fit_params, cam_rec.get_matrix(), lmks.cpu().float())
        
            x0 = torch.cat([0*torch.rand(pfs_f.num_components), torch.tensor(tau), torch.tensor(u)]).to(device)
            x0, fit_out = pfd_f.fit_GN(x0.float(), [torch.from_numpy(pts).to(device).float()], 
                                       plotit=False, plotlast=False, use_ineq = True)
            alphas.append(x0[:pfs_f.num_components])
            taus.append(x0[pfs_f.num_components:pfs_f.num_components+3])
            us.append(x0[pfs_f.num_components+3:pfs_f.num_components+6])
            
            all_pts.append(torch.from_numpy(pts).to(device).float())
            # break
        # return
        
        idx = np.arange(len(taus)).astype(int)
        random.shuffle(idx)
        idx = idx[:Nframes]
        calphas = [alphas[x] for x in idx]
        ctaus = [taus[x] for x in idx]
        cus = [us[x] for x in idx]
        call_pts = [all_pts[x] for x in idx]
        
        pfd = models.perspective_fitter.PerspectiveFitter(mm, cam_rec, use_maha=True, 
                                                          which_pts=which_pts, 
                                                          F=Nframes)
        x0 = pfd.construct_initialization(calphas, ctaus, cus)
        x, fit_out = pfd.fit_GN(x0.float(), call_pts, plotit=False, plotlast=False, use_ineq = True)
        
        alpha = x[:pfd.num_components].reshape(1,-1)
        
        
        p0 = mm.compute_face_shape(alpha).squeeze().cpu().numpy()
        p0s.append(p0)
    p0 = np.mean(p0s, axis=0)
    # m = {'pts': p0}
    # scipy.io.savemat(tpath, m)
    np.savetxt(tpath_txt, p0)

    
    
    
    

def compute_neutral_face0(mm, all_shapes_persubj, subj_id):
    
    fov = 20    
    alpha_path = f'{outdir_final}/{subj_id}.alpha'
    
    if os.path.exists(alpha_path):
        return np.loadtxt(alpha_path)
    
    random.seed(1907)
    
    Nframes = 22
    cshapes = []

    for shapei in range(min(len(all_shapes_persubj), Nframes)):
        
        cshapes.append(all_shapes_persubj[shapei])
    
    
    tdir = f'/offline_data/face/synth/identity/output/fov020/3DIDv0-M{Nframes:02d}'
    os.makedirs(tdir, exist_ok=True)
    tpath_txt = f'{tdir}/id{subj_id:03d}_001.txt'
    
    np.savetxt(tpath_txt, np.mean(cshapes, axis=0))

    

for subj_id in range(0,100):
    print(subj_id)
    compute_neutral_face_multi(mm, all_pts_persubj[subj_id], all_invMs_persubj[subj_id], all_ims_persubj[subj_id], subj_id)


#%%
for subj_id in range(0,100):
    print(subj_id)
    compute_neutral_face(mm, all_pts_persubj[subj_id], all_invMs_persubj[subj_id], all_ims_persubj[subj_id], subj_id)


#%%
for subj_id in range(0,100):
    print(subj_id)
    compute_neutral_face0(mm, all_shapes_persubj[subj_id], subj_id)


