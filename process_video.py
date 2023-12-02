#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:37:06 2023

@author: sariyanide
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import copy
from utils import utils

import os
from math import floor, ceil
import models.medium_model
import models.morphable_model
import models.separated_model

import models.orthograhic_fitter
import models.perspective_fitter
from torchvision.transforms import Grayscale

device = 'cuda'

fov = 15
rasterize_size = 224
cx = rasterize_size/2
rdir = '/online_data/face/yt_faces2/yt_cropped2'#sys.argv[2]
# rdir = '/online_data/face/yt_faces/yt_cropped'#sys.argv[2]
which_model = 'sep_model'

dbname = os.path.basename(rdir)

import sys
sys.path.append('./models')
import models

checkpoint_dir = 'checkpoints_norm'

which_bfm = 'BFMmm-23660'

cfgid =2
Ntra = 139979
learning_rate = 1e-5
backbone = 'resnet50'
tform_data= True

init_path_id = f'{checkpoint_dir}/medium_model15.00combined_celeb_ytfacesresnet50{Ntra}True1e-05-{cfgid}-BFMmm-23660UNL_STORED.pth'

checkpoint_id = torch.load(init_path_id)
model_id = models.medium_model.MediumModel(rasterize_fov=fov, rasterize_size=rasterize_size, 
                                        label_means=checkpoint_id['label_means'].to(device), 
                                        label_stds=checkpoint_id['label_stds'].to(device),
                                        which_bfm=which_bfm, which_backbone=backbone)

model_id.load_state_dict(checkpoint_id['model_state'])
model_id.to(device)
model_id.eval()

mm = models.morphable_model.MorphableModel(key=which_bfm, device=device)

# init_path_id = 'checkpoints_norm/medium_model5.00yt_cropped2resnet18BFMmm-23660.pth'
init_path_perframe = init_path_id
# checkpoint = torch.load(f'{checkpoint_dir}/{which_model}{fov:.2f}{dbname}.pth')

spath = f'{checkpoint_dir}/sep_modelv3SP15.00combined_celeb_ytfacesresnet501e-05{cfgid}True{Ntra}_V2.pth'

checkpoint = torch.load(spath)
model = models.separated_model.SeparatedModelV3(rasterize_fov=fov, rasterize_size=rasterize_size,
                                        label_means=checkpoint_id['label_means'].to(device), 
                                        label_stds=checkpoint_id['label_stds'].to(device),
                                        init_path_id=init_path_id,
                                        init_path_perframe=init_path_perframe,
                                        which_backbone=backbone)

model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()


vpath = '/offline_data/face/CAR/InfantSync_3DI/preprocessing/id-IS002_m-tc_cond-is_adm-001_n-20200203110206_2_undistorted.mp4'
lmkspath = '/offline_data/face/CAR/InfantSync_3DI/preprocessing/id-IS002_m-tc_cond-is_adm-001_n-20200203110206_2_undistorted.landmarks.global4'

# vpath = '/home/sariyanide/code/3DMMD/ML0001_2.mp4'
# lmkspath = vpath.replace('mp4', 'csv')
# vpath = '/offline_data/face/CAR/InfantSync/videos/id-IS015_m-tc_cond-is_adm-001_n-20201119131510_2.mp4'
# csvpath = '/offline_data/face/CAR/InfantSync/videos/id-IS015_m-tc_cond-is_adm-001_n-20201119131510_2.csv'

f_x_treecam = 6.7210446166992188e+02
f_y_treecam = 6.3589253234863281e+02
cx_treecam = 7.0773209823712750e+02
cy_treecam = 9.1724920068352912e+02


cam_3DID = models.camera.Camera(fov, fov, cx, cx)  


#%%


use_exp_model = True
from utils import utils

def process_video(vpath, lmkspath, cam):
    # cam = models.camera.Camera(fov, fov, cx, cx)    
    
    if lmkspath.find('.csv') >= 0:
        csv = pd.read_csv(lmkspath)
        L = csv.values[:,1:]
    else:
        L = np.loadtxt(lmkspath)
    
    all_params_path = vpath.replace('.mp4', f'-{use_exp_model}-3Dparams.npy')
    
    if os.path.exists(all_params_path):
        return np.load(all_params_path, allow_pickle=True).item()
    
    cap = cv2.VideoCapture(vpath)
    
    transform_gray = Grayscale(num_output_channels=3)

    alphas = []
    exps = []
    angles = []
    betas = []
    taus = []
    invMs = []
    frame_idx = 0
    
    while(True):    
        print('\rProcessing frame %d/%d'%(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), end="")
        frame_idx += 1
        ret, frame = cap.read()
    
        if not ret:
            break 
        
        lmks51 = L[frame_idx-1,:].reshape(-1,2).astype(np.float32)
        
        if lmks51.shape[0] == 68:
            lmks51 = lmks51[17:,:]
        
        if lmks51[0,0] == 0:
            alphas.append(alphas[-1])
            betas.append(betas[-1])
            angles.append(angles[-1])
            taus.append(taus[-1])
            exps.append(exps[-1])
            invMs.append(invMs[-1])
            continue
        
        cim = frame.astype(np.float32)/255.0
        # lmks51[:,1] = cim.shape[0]-lmks51[:,1]
        M = utils.estimate_norm(lmks51, cim.shape[0], 1.5, [25,25])
        cim = utils.resize_n_crop_cv(cim, M, rasterize_size)
        invM = utils.estimate_inv_norm(lmks51, 1920, 1.5, [25,25])
        lmks51_hom = np.concatenate((lmks51, np.ones((lmks51.shape[0], 1))), axis=1)
        
        """
        lmks_new = (M @ lmks51_hom.T).T
        icim = utils.resize_n_crop_inv_cv(cim, invM, (frame.shape[1], frame.shape[0]))
        if False and frame_idx == 30:
            print(M)
            print(invM)
            plt.figure(figsize=(50, 70))
            plt.imshow(frame)
            plt.plot(lmks51_hom[:,0], lmks51_hom[:,1])
            plt.show()
            plt.imshow(cim)
            plt.plot(lmks_new[:,0], lmks_new[:,1])
            plt.show()
            # print(frame)
            print(icim.shape)
            plt.figure(figsize=(50, 70))
            plt.imshow(icim)
            plt.show()
            # break
        """
    
        cim = np.transpose(cim, (2,0,1))
        cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
        cim = cim.to(device)
        
        if not use_exp_model:
            y = model_id(cim)
            params = model_id.parse_params(y[0])
        else:
            y, alpha_un, beta_un, _ = model.forward(cim)
            params = model.parse_params(y, alpha_un, beta_un)

        """
        if frame_idx == 200:
            pts = mm.project_to_2d(cam, params['angles'], params['tau'], params['alpha'], params['exp']).detach().cpu().squeeze().numpy()
            pts[:,1] = rasterize_size-pts[:,1]
            print(pts.shape)
            pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
            pts = (invM @ pts.T).T
            plt.figure(figsize=(1.4*50, 1.4*70))
            plt.imshow(frame)
            plt.plot(pts[:,0], pts[:,1], '.')
            plt.savefig('c.jpg')
            break
        """

        (mask, _, rim), pr = model_id.render_image(params)
        mask = mask.detach().cpu().numpy()[0,0,:,:]
        cim0 = cim.detach().cpu().numpy()[0,0,:,:]
        rim = rim.detach().cpu().numpy()[0,0,:,:]
        
        rim[mask==0] = (cim0[mask==0])
        rim[mask==1] = (cim0[mask==1]+2*rim[mask==1])/3.0
                
        """
        if frame_idx % 2  == 0:
            p = mm.compute_face_shape(params['alpha'], params['exp'])
            p = p.detach().cpu().squeeze().numpy()
            
            # plt.clf()
            plt.figure(frame_idx, figsize=(30*1.5,10*1.5))
            
                
            plt.subplot(141)
            plt.imshow(cim0)
            
            plt.subplot(142)
            plt.imshow(rim)
            
            plt.subplot(143)
            # plt.plot(p0[:,0], p0[:,1], '.')
            plt.plot(p[:,0], p[:,1], '.')
            plt.ylim((-90, 90))
            
            
            plt.subplot(144)
            # plt.plot(p0[:,2], p0[:,1], '.')
            plt.plot(p[:,2], p[:,1], '.')
            plt.ylim((-90, 90))
            plt.show()
            """
        
        
        
        alphas.append(params['alpha'].detach().cpu())
        betas.append(params['beta'].detach().cpu())
        angles.append(params['angles'].detach().cpu())
        taus.append(params['tau'].detach().cpu())
        exps.append(params['exp'].detach().cpu())
        invMs.append(invM)
        
        # if frame_idx == 1800:
            # break
        
        
    all_params = {'alphas': alphas,
                  'betas': betas,
                  'angles': angles,
                  'taus': taus,
                  'exps': exps,
                  'invMs': invMs
                  }
    np.save(all_params_path, all_params)
    
    
    return all_params




all_params = process_video(vpath, lmkspath, cam_3DID)
mm = models.morphable_model.MorphableModel(key=which_bfm, device=device)
cam_tree = models.camera.Camera(f_x=f_x_treecam, f_y=f_y_treecam, cx=cx_treecam, cy=cy_treecam)

T = len(all_params['alphas'])





#%%
from scipy.spatial import distance
import models.perspective_fitter
import random
random.seed(1907)

def compute_neutral_face(all_params, random_seed=0, F0=7, use_exp=False):
    of = models.orthograhic_fitter.OrthographicFitter(mm)
    
    pfs_f = models.perspective_fitter.PerspectiveFitter(mm, cam_tree, use_maha=True, which_pts='lmks', F=1)
    pfd_f = models.perspective_fitter.PerspectiveFitter(mm, cam_tree, use_maha=True, which_pts='sampled', F=1)
    
    
    T = len(all_params['taus'])
    alphas = []
    taus = []
    us = []
    
    all_pts = []
    
    for t in range(0, T, 120):
        print(f'\rt={t}/T', end='')
        alpha = all_params['alphas'][t].to(mm.device)
        exp = all_params['exps'][t].to(mm.device)
        tau = all_params['taus'][t].to(mm.device)
        angles = all_params['angles'][t].to(mm.device)
        invM = all_params['invMs'][t]
        if use_exp:
            pts = mm.project_to_2d(cam_3DID, angles, tau, alpha, exp).detach().cpu().squeeze().numpy()
        else:
            pts = mm.project_to_2d(cam_3DID, angles, tau, alpha, exp*0).detach().cpu().squeeze().numpy()
            
        pts[:,1] = rasterize_size-pts[:,1]
        # print(pts.shape)
        pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
        pts = (invM @ pts.T).T
        # pts[:,1] = 1920-pts[:,1]
        # pts[:,1] = 224-pts[:,1]
        """
        if t % 3 == 0:
            plt.figure(figsize=(14.4*5,  19.2*5))
            plt.plot(pts[:,0], -pts[:,1], '.')
            plt.xlim((0, 1440))
            plt.ylim((-1920, 0))
            plt.show()
        continue
        """
        lmks = torch.from_numpy(pts).to(mm.device)[mm.li,:]
        # lmks[:,1] = 1920-lmks[:,1]
        of_fit_params = of.fit_orthographic_GN(lmks.cpu().numpy(), plotit=False)[0]
        u = torch.tensor(of_fit_params['u'])
        tau = of.to_projective(of_fit_params, cam_tree.get_matrix(), lmks.cpu().float())
    
        x0 = torch.cat([0*torch.rand(pfs_f.num_components), torch.tensor(tau), torch.tensor(u)]).to(mm.device)
        # x0 = pfd_f.fit_GN(x0.float(), [torch.from_numpy(pts).to(mm.device).float()], plotit=True, use_ineq = False)[0]
        # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = True)[0]
        # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = False)[0]
        # break
        # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = True)[0]
        x0 = pfd_f.fit_GN(x0.float(), [torch.from_numpy(pts).to(mm.device).float()], plotit=False, plotlast=False, use_ineq = True)[0]
    
        alphas.append(x0[:pfs_f.num_components])
        taus.append(x0[pfs_f.num_components:pfs_f.num_components+3])
        us.append(x0[pfs_f.num_components+3:pfs_f.num_components+6])
        
        all_pts.append(torch.from_numpy(pts).to(mm.device).float())
    
    idx = np.arange(len(taus)).astype(int)[:F0]
    alphas = [alphas[x] for x in idx]
    taus = [taus[x] for x in idx]
    us = [us[x] for x in idx]
    # taus = taus[idx]
    # us = us[idx]
    # F0 = len(taus)
    # F = len(taus[:F0])
    pfd = models.perspective_fitter.PerspectiveFitter(mm, cam_tree, use_maha=True, which_pts='sampled', F=F0)
    
    x0 = pfd.construct_initialization(alphas[:F0], taus[:F0], us[:F0])
    x = pfd.fit_GN(x0.float(), all_pts, plotit=False, plotlast=False, use_ineq = True)[0]
    
    return x[:pfd.num_components]


alpha1 = compute_neutral_face(all_params, use_exp=True)
alpha2 = compute_neutral_face(all_params, use_exp=False)


#%%
p1 = mm.compute_face_shape(alpha1.unsqueeze(0)).cpu().squeeze()
p2 = mm.compute_face_shape(alpha2.unsqueeze(0)).cpu().squeeze()

#%%

plt.figure(figsize=(40,70))
plt.subplot(321)
plt.plot(p1[:,0], p1[:,1], '.')

plt.subplot(322)
plt.plot(p1[:,2], p1[:,1], '.')


plt.subplot(323)
plt.plot(p2[:,0], p2[:,1], '.')

plt.subplot(324)
plt.plot(p2[:,2], p2[:,1], '.')


ix = mm.li.cpu()
ms = 18
plt.subplot(325)
plt.plot(p1[ix,0], p1[ix,1], '.', markersize=ms)
plt.plot(p2[ix,0], p2[ix,1], '.', markersize=ms)

plt.subplot(326)
plt.plot(p1[ix,2], p1[ix,1], '.', markersize=ms)
plt.plot(p2[ix,2], p2[ix,1], '.', markersize=ms)
#%%
from time import time

plt.figure(figsize=(80,40))

plt.subplot(121)
plt.plot(p1[:,0], p1[:,1], '.', markersize=ms)
plt.plot(p2[:,0], p2[:,1], '.', markersize=ms)

plt.subplot(122)
plt.plot(p1[:,2], p1[:,1], '.', markersize=ms)
plt.plot(p2[:,2], p2[:,1], '.', markersize=ms)


def compute_expression_coeffs(alpha, Tmax=260):
    mm_alpha = models.morphable_model.MorphableModel(key=which_bfm, device=device)
    mm_alpha.update_mean_face(alpha)
    
    of = models.orthograhic_fitter.OrthographicFitter(mm_alpha)
    
    pfs = models.perspective_fitter.PerspectiveFitter(mm_alpha, cam_tree, use_maha=False, which_pts='lmks', F=1, which_basis='expression')
    pfd = models.perspective_fitter.PerspectiveFitter(mm_alpha, cam_tree, use_maha=False, which_pts='sampled', F=1, which_basis='expression')
    
    
    us = []
    taus = []
    exps = []
    poses = []
    exp_prev = None
    u_prev = None
    tau_prev = None
    
    for t in range(Tmax):
        print('\rProcessing frame %d/%d' % (t, T), end="")
    
        angles = all_params['angles'][t].to(mm.device)
        tau = all_params['taus'][t].to(mm.device)
        alpha = all_params['alphas'][t].to(mm.device)
        exp = all_params['exps'][t].to(mm.device)
        pts = mm.project_to_2d(cam_3DID, angles, tau, alpha, exp).cpu().squeeze().to(mm_alpha.device)
        pts[:,1] = rasterize_size-pts[:,1]
        pts = pts.cpu().numpy()
        pts = np.concatenate((pts, torch.ones((pts.shape[0], 1))), axis=1)
        invM = all_params['invMs'][t]
        
        pts = (invM @ pts.T).T
    
        lmks = pts[mm_alpha.li.cpu(),:]# .cpu().numpy()
        
        if len(exps) == 0:
            of_fit_params = of.fit_orthographic_GN(lmks, plotit=False)[0]
            u = torch.tensor(of_fit_params['u'])
            tau = of.to_projective(of_fit_params, cam_tree.get_matrix(), lmks)
            x0 = torch.cat([0*torch.rand(pfd.num_components), torch.tensor(tau), u]).to(mm_alpha.device)
        # else:
            # x0 = torch.cat([torch.from_tensor(exps[-1]).to(mm_alpha.device), torch.tensor(tau), u])
            
        # x0 = pfs.fit_GN(x0.float(), [pts.cpu().numpy()], plotit=True, use_ineq = False)[0]
        # x0 = pfd.fit_GN(x0.float(), [pts.cpu().numpy()], plotit=True, use_ineq = False)[0]
        # t1 = time()
        
        # plotlast = t % 30 == 0
        x0 = pfd.fit_GN(x0.float(), [torch.from_numpy(pts).to(pfd.mm.device).float()], plotit=False, 
                        use_ineq = True, plotlast=False)[0]
        # print(time()-t1)
        exps.append(x0[:pfs.num_components].cpu().numpy())
        tau = x0[pfs.num_components:pfs.num_components+3].cpu().numpy()
        u = x0[pfs.num_components+3:pfs.num_components+6].cpu().numpy()
        
        taus.append(tau)
        us.append(u)
        
        pose = np.concatenate((tau, u), axis=0)
        
        poses.append(pose)
        # break
        if t == 600:
            break

    return exps, poses
#%%


exps, poses = compute_expression_coeffs(alpha1)
#%%
T = len(exps)

orig_vidpath = '/offline_data/face/CAR/InfantSync/id-IS002/sess-001/id-IS002_m-tc_cond-is_adm-001_n-20200203110206_2.mp4'

tex = np.loadtxt('/home/sariyanide/code/3DI/build/models/MMs/BFMmm-23660/tex_mu.dat').reshape(-1,1)
bn = os.path.basename(vpath).replace('.mp4', '')
illums = np.tile([48.06574, 9.913327, 798.2065, 0.005], (T, 1))
ddir = '/offline_data/tmp/vids/'
shpsm_fpath = f'{ddir}/{bn}.shapesm'
tex_fpath = f'{ddir}/{bn}.betas'
illums_fpath = f'{ddir}/{bn}.illums'
poses_fpath = f'{ddir}/{bn}.poses'
exps_fpath = f'{ddir}/{bn}.expressions'
cfg_fpath = f'/home/sariyanide/car-vision/cuda/build-3DI/configs/{which_bfm}.cfg1.global4.txt'
render3ds_path = f'{ddir}/{bn}-{use_exp_model}_S10.avi' 
texturefs_path = f'{ddir}/{bn}_texture_sm.avi' 

# alphas = alphas.detach().cpu().numpy().reshape(-1,1)
# betas = betas.detach().cpu().numpy().reshape(-1,1)

exps = np.array(exps)



poses_np = np.array(poses)


# taus[1] *= -1
# us[0] *= -1
# us[-1] *= -1


# poses = np.zeros(poses.shape)
# poses_np[:,3] = 0.
# poses_np[:,4] = 0.11
poses_np[:,2] -= 300
# poses_np[:,3] *= -1
# poses_np[:,3] -= np.pi
poses_np[:,-1] *= -1
# poses[:,3] *= -1
# poses[:,4] *= -1
# poses[:,-1] *= -1
p0= mm.compute_face_shape(alpha2.unsqueeze(0))
p0 = p0.detach().squeeze()
p0[:,1] *= -1
# p0[:,2] += 20
# p0[:,2] = max(p0[:,2])-p0[:,2]

np.savetxt(shpsm_fpath, p0.cpu().numpy())
np.savetxt(tex_fpath,tex)
np.savetxt(exps_fpath, exps)
np.savetxt(poses_fpath, poses_np)
np.savetxt(illums_fpath, illums)


# pose = np.tile([6.9, 43., 799.,-0.16, -0.39, 0.125, -0.38, -0.18, 0.16], (T, 1))
# pose = np.tile([3, 43., 799.,0, 0,0.01, -0.38, -0.18, 0.16], (T, 1))

# os.chdir('/home/sariyanide/car-vision/cuda/build-3DI')

cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % (orig_vidpath, cfg_fpath, 'models/cameras/TreeCam_1041a.txt', shpsm_fpath, tex_fpath,
                                                                   exps_fpath, poses_fpath, illums_fpath, 
                                                                   render3ds_path, texturefs_path)

print('\n')
print(cmd_vis)
    








    
