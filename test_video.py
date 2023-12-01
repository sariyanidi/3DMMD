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
from torchvision.transforms import Grayscale

device = 'cuda'

fov = 15
rasterize_size = 224
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
Ntot = 81827
learning_rate = 1e-5
backbone = 'resnet50'
tform_data= True
init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{Ntot}{tform_data}{learning_rate}{which_bfm}.pth'
# init_path_id = f'{checkpoint_dir}/medium_model{fov:.2f}{dbname}{backbone}{which_bfm}_02381.pth'

# init_path_id = f'{checkpoint_dir}/medium_model20.00combined_celeb_ytfacesresnet50BFMmm-23660_01711.pth'
# init_path_id = f'{checkpoint_dir}/medium_model20.00combined_celeb_ytfacesresnet5081866True1e-05BFMmm-23660.pth'

init_path_id = f'{checkpoint_dir}/medium_model15.00combined_celeb_ytfacesresnet50351639True1e-05-{cfgid}-BFMmm-23660UNL.pth'

checkpoint_id = torch.load(init_path_id)
model_id = models.medium_model.MediumModel(rasterize_fov=fov, rasterize_size=rasterize_size, 
                                        label_means=checkpoint_id['label_means'].to(device), 
                                        label_stds=checkpoint_id['label_stds'].to(device),
                                        which_bfm=which_bfm, which_backbone=backbone)

model_id.load_state_dict(checkpoint_id['model_state'])
model_id.to(device)
model_id.eval()

mm = models.morphable_model.MorphableModel(key=which_bfm, device=device)



#%%

# init_path_id = f'checkpoints_norm/medium_model30.00{dbname}.pth'
# init_path_perframe = f'checkpoints_norm/medium_model30.00{dbname}.pth'

# init_path_id = 'checkpoints_norm/medium_model5.00yt_cropped2resnet18BFMmm-23660.pth'
init_path_perframe = init_path_id
# checkpoint = torch.load(f'{checkpoint_dir}/{which_model}{fov:.2f}{dbname}.pth')

# medium_model20.00combined_celeb_ytfacesresnet501True1e-05BFMmm-23660.pth


checkpoint = torch.load(f'checkpoints_norm/sep_modelv3SP15.00combined_celeb_ytfacesresnet501e-05{cfgid}True139979.pth')
model = models.separated_model.SeparatedModelV3(rasterize_fov=fov, rasterize_size=rasterize_size,
                                        label_means=checkpoint_id['label_means'].to(device), 
                                        label_stds=checkpoint_id['label_stds'].to(device),
                                        init_path_id=init_path_id,
                                        init_path_perframe=init_path_perframe,
                                        which_backbone=backbone)

model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()
"""

"""



#%%



def crop_frame(frame, lmks):

    # lmks = np.loadtxt(lf)
    xs = lmks[:,0]
    ys = lmks[:,1]
    xmin = floor(xs.min())
    xmax = ceil(xs.max())
    ymin = floor(ys.min())
    ymax = ceil(ys.max())

    w = xmax-xmin
    h = ymax-ymin
    s = max(w, h)

    xmin -= int(s/3)
    xmax += int(s/3)
    ymin -= int(s/3)
    ymax += int(s/3)
    w = xmax-xmin
    h = ymax-ymin
    s = max(w, h)

    xs = xs.reshape(-1,1)
    ys = ys.reshape(-1,1)
    
    yoff = -int(s*0.1)
    
    # r = cv2.Rect(xmin, ymin, w, h)
    im = frame[ymin+yoff:ymin+s+yoff, xmin:xmin+s, :]
    cs = im.shape[0]
    im = cv2.resize(im, (224, 224))
    lmks = np.concatenate((xs-xmin, ys-ymin), axis=1)
    lmks *= 224.0/cs
    
    return im, lmks

vpath = '/home/sariyanide/code/3DMMD/ML0001_2.mp4'
# vpath = '/home/sariyanide/code/3DMMD/out.mp4'
csvpath = vpath.replace('mp4', 'csv')
# vpath = '/offline_data/face/CAR/InfantSync/videos/id-IS015_m-tc_cond-is_adm-001_n-20201119131510_2.mp4'
# csvpath = '/offline_data/face/CAR/InfantSync/videos/id-IS015_m-tc_cond-is_adm-001_n-20201119131510_2.csv'
csv = pd.read_csv(csvpath)

L = csv.values[:,1:]
cap = cv2.VideoCapture(vpath)


transform_gray = Grayscale(num_output_channels=3)


betas_un = []
alphas_un = []
betas = []
alphas = []
frame_idx = 0
while(True):    
    print('\rProcessing frame %d/%d'%(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), end="")

    frame_idx += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break 

    if frame_idx % 20 != 0:
        continue
    
    lmks68 = L[frame_idx-1,:].reshape(-1,2).astype(np.float32)
    # frame, lmks68 = crop_frame(frame, lmks68)
    
    
    cim = frame.astype(np.float32)/255.0
    lmks68[:,1] = cim.shape[0]-lmks68[:,1]
    M = utils.estimate_norm(lmks68[17:,:], cim.shape[0], 1.5, [25,25])
    cim = utils.resize_n_crop_cv(cim, M, rasterize_size)
    cim = np.transpose(cim, (2,0,1))
    # plt.imshow(cim[0,:,:])
    # break
    cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
    cim = cim.to(device)
    
    y = model_id(cim)
    params_un = model_id.parse_params_unnormalized(y[0])
    params = model_id.parse_params(y[0])
    
    (mask, _, rim), pr = model_id.render_image(params)
    mask = mask.detach().cpu().numpy()[0,0,:,:]
    cim0 = cim.detach().cpu().numpy()[0,0,:,:]
    rim = rim.detach().cpu().numpy()[0,0,:,:]
    
    rim[mask==0] = (cim0[mask==0])
    rim[mask==1] = (cim0[mask==1]+2*rim[mask==1])/3.0
    

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
    alphas_un.append(params_un['alpha'])
    betas_un.append(params_un['beta'])
    alphas.append(params['alpha'])
    betas.append(params['beta'])
    
    if frame_idx >= 600:
        break



alpha = torch.cat(alphas).mean(dim=0).unsqueeze(0)
beta = torch.cat(betas).mean(dim=0).unsqueeze(0)
alpha_un = torch.cat(alphas_un).mean(dim=0).unsqueeze(0)
beta_un = torch.cat(betas_un).mean(dim=0).unsqueeze(0)

plt.show()
# break






#%%


from time import time


cap = cv2.VideoCapture(vpath)

os.makedirs('figures', exist_ok=True)
exps = []
poses = []
illums = []
frame_idx = 0
while(True):    
    print('\rProcessing frame %d/%d'%(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), end="")

    frame_idx += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break 
    
    if frame_idx < 1160:
        continue
    # if frame_idx % 3 != 0:
        # continue
    

    # frame, lmks68 = crop_frame(frame, lmks68)
    
    
    # cim = frame.astype(np.float32)/255.0
    # lmks68[:,1] = cim.shape[0]-lmks68[:,1]
    # M = utils.estimate_norm(lmks68[17:,:], cim.shape[0], 1.5, [25,25])
    # cim = utils.resize_nexps_crop_cv(cim, M, rasterize_size).astype(np.float32)/255.0
    # cim = np.transpose(cim, (2,0,1))


    # cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
    
    
    cexps = []
    ctaus = []
    cangles = []
    t0 = time()

    for nrep in range(1):
        lmks68 = L[frame_idx-1,:].reshape(-1,2).astype(np.float32)
        cim = copy.deepcopy(frame).astype(np.float32)/255.0
        lmks68[:,1] = cim.shape[0]-lmks68[:,1]
        M = utils.estimate_norm(lmks68[17:,:]+0*np.random.randn(51,2), cim.shape[0], 1.5, [25,25])
        cim = utils.resize_n_crop_cv(cim, M, rasterize_size)
        cim = np.transpose(cim, (2,0,1))
        cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
    
        cim = cim.to(device)
        
        # y = model_id(cim)[0]
        # params = model_id.parse_params(y)
        # 
        y, alpha_un, beta_un, _ = model.forward(cim)
        paramstmp = model.parse_params(y, alpha_un, beta_un)
        y = model.test(cim, paramstmp['alpha'], paramstmp['beta'])
        # y, alpha_un, beta_un, _ = model.forward(cim)
        params = model.parse_params(y, alpha_un, betas_un[10])
        cexps.append(params['exp'].detach().squeeze().cpu().numpy())
        cexps.append(params['exp'].detach().squeeze().cpu().numpy())
        ctaus.append(params['tau'].detach().squeeze().cpu().numpy())
        ctaus.append(params['tau'].detach().squeeze().cpu().numpy())
        cangles.append(params['angles'].detach().squeeze().cpu().numpy())
        cangles.append(params['angles'].detach().squeeze().cpu().numpy())
        
    tf = time()
    print('Took ', (tf-t0))
    # plt.clf()

    
    # p = mm.compute_face_shape( alpha, params['exp'])
    p = mm.compute_face_shape( params['alpha'], params['exp'])
    p = p.detach().cpu().squeeze().numpy()

    plt.figure(frame_idx, figsize=(45*0.65,10*0.65))
    (mask, _, rim), pr = model.render_image(params)

    # plt.plot(params['exp'][0].detach().cpu(), alpha=0.4)
    plt.subplot(151)
    plt.imshow(cim.detach().cpu().numpy()[0,0,:,:])
    
    plt.subplot(152)
    plt.imshow(cim.detach().cpu().numpy()[0,1,:,:])
    
    plt.subplot(153)
    plt.imshow(rim.detach().cpu().numpy()[0,0,:,:])
    
    
    plt.subplot(154)
    # plt.plot(p0[:,2], p0[:,1], '.')
    plt.plot(p[:,0], p[:,1], '.')
    plt.ylim((-90, 90))
    
    plt.subplot(155)
    # plt.plot(p0[:,2], p0[:,1], '.')
    plt.plot(p[:,2], p[:,1], '.')
    plt.ylim((-90, 90))
    plt.savefig(f'figures/{frame_idx:04d}.jpg', bbox_inches='tight')
    plt.show()
    """
    """
    cexps = np.array(cexps)
    ctaus = np.array(ctaus)
    cangles = np.array(cangles)
    angles = np.mean(cangles,axis=0)
    tau = np.mean(ctaus,axis=0)
    pose = np.concatenate((tau, angles), axis=0)
    
    exps.append(np.mean(cexps, axis=0))
    poses.append(pose)
    
    if frame_idx == 1600:
        break
    # break

#%%

tex = np.loadtxt('/home/sariyanide/code/3DI/build/models/MMs/BFMmm-23660/tex_mu.dat').reshape(-1,1)
T = frame_idx
bn = os.path.basename(vpath)
illums = np.tile([48.06574, 9.913327, 798.2065, 0.005], (T, 1))
ddir = '/offline_data/tmp/vids/'
shpsm_fpath = f'{ddir}/{bn}.shapesm'
tex_fpath = f'{ddir}/{bn}.betas'
illums_fpath = f'{ddir}/{bn}.illums'
poses_fpath = f'{ddir}/{bn}.poses'
exps_fpath = f'{ddir}/{bn}.expressions'
cfg_fpath = f'/home/sariyanide/car-vision/cuda/build-3DI/configs/{which_bfm}.cfg1.global4.txt'
render3ds_path = f'{ddir}/{bn}_3D_pose-exp-greenbg.avi' 
texturefs_path = f'{ddir}/{bn}_texture_sm.avi' 

# alphas = alphas.detach().cpu().numpy().reshape(-1,1)
# betas = betas.detach().cpu().numpy().reshape(-1,1)

exps = np.array(exps)

poses = np.array(poses)
p0= mm.compute_face_shape(alpha, 0*params['exp'])
p0 = p0.detach().squeeze()
p0[:,1] *= -1
p0[:,2] += 20
# p0[:,2] = max(p0[:,2])-p0[:,2]

np.savetxt(shpsm_fpath, p0.cpu().numpy())
np.savetxt(tex_fpath,tex)
np.savetxt(exps_fpath, exps)
np.savetxt(poses_fpath, poses)
np.savetxt(illums_fpath, illums)


# pose = np.tile([6.9, 43., 799.,-0.16, -0.39, 0.125, -0.38, -0.18, 0.16], (T, 1))
# pose = np.tile([3, 43., 799.,0, 0,0.01, -0.38, -0.18, 0.16], (T, 1))
    

cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % (vpath, cfg_fpath, str(fov), shpsm_fpath, tex_fpath,
                                                                   exps_fpath, poses_fpath, illums_fpath, 
                                                                   render3ds_path, texturefs_path)

print('\n')
print(cmd_vis)
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

"""
if args.save_result_video:
    cap_result.release()
    print('Saved result video to %s' % out_vidpath)
    """
    
    
