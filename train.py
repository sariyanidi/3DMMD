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
from glob import glob
# sys.path.append('../')

from scipy.spatial.transform import Rotation
from utils import utils
from models import camera

import matplotlib.pyplot as plt
from models import morphable_model, orthograhic_fitter, simple_model, medium_model

from torchvision.transforms import Grayscale
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from time import time



class SimpleDataset(Dataset):
    
    def __init__(self, rootdir='/media/v/SSD1TB/dataset/for_3DID/cropped/', is_train=True,
                 transform=None, normalize_labels=False,
                 fov=12.56, rasterize_size=224):
        self.rootdir = rootdir
        # self.extn = 'label0000'
        self.extn = 'label_eulerrod2'
        self.normalize_labels = normalize_labels
        all_label_paths = glob(f'{self.rootdir}/*{self.extn}')
        all_label_paths.sort()
        
        self.transform = transform
        self.fov = fov
        self.rasterize_size = rasterize_size
        
        Ntot = len(all_label_paths)
        Ntra = int(0.75*Ntot)
        
        self.A = None
        if self.normalize_labels:
            all_labels = []
            for f in all_label_paths[:Ntra]:
                all_labels.append(np.loadtxt(f))
            self.A = np.array(all_labels)
            self.stds = np.std(self.A, axis=0)
            self.means = np.mean(self.A, axis=0)
            # print(A)
        if is_train:
            self.label_paths = all_label_paths[:Ntra]
        else:
            self.label_paths = all_label_paths[Ntra:]
        

    def __len__(self):
        return len(self.label_paths)


    def __getitem__(self, idx):
        label_fpath = self.label_paths[idx]
        img_fpath = label_fpath.replace(f'.{self.extn}', '.jpg')
        lmks_fpath = label_fpath.replace(f'.{self.extn}', '.txt')

        image = read_image(img_fpath)
        label = np.loadtxt(label_fpath)
        # label[-1] -= 1000
        lmks = np.loadtxt(lmks_fpath)
        tlmks = copy.deepcopy(lmks)
        tlmks[:,1] = self.rasterize_size-tlmks[:,1]
        rigid_tform = utils.estimate_norm(tlmks[17:,:], self.rasterize_size)
        
        if self.transform:
            image = self.transform(image)
        if self.normalize_labels:
            label -= self.means
            label /= self.stds
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image.float(), \
            torch.from_numpy(label).float(), \
                torch.from_numpy(lmks).float(), \
                    torch.from_numpy(rigid_tform).float()
    
    def get_with_lmks(self, idx):
        label_fpath = self.label_paths[idx]
        lmks_fpath = label_fpath.replace(f'.{self.extn}', '.txt')
        img_fpath = label_fpath.replace(f'.{self.extn}', '.jpg')
        image = read_image(img_fpath)
        label = np.loadtxt(label_fpath)
        lmks = np.loadtxt(lmks_fpath)
        
        rigid_tform = utils.estimate_norm(lmks, self.rasterize_size)
        # label[-1] -= 1000
        if self.transform:
            image = self.transform(image)
        if self.normalize_labels:
            label -= self.means
            label /= self.stds
        
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image.float().unsqueeze(0), \
                    torch.from_numpy(label).float().unsqueeze(0), \
                    torch.from_numpy(lmks).float().unsqueeze(0), \
                        torch.from_numpy(rigid_tform).float().unsqueeze(0)
                        
    
    
    def create_labels(self, model):
        
        cam = camera.Camera(self.fov, self.fov, 
                            self.rasterize_size, self.rasterize_size)

        fitter = orthograhic_fitter.OrthographicFitter(model)
        
        files = glob(f'{self.rootdir}/*txt')
        files.sort()
        
        Y = []
        
        for f in files:
            # print(f)
            label_path = f.replace('txt', self.extn)
            
            if os.path.exists(label_path):
                continue
            
            lmks = np.loadtxt(f)[17:,:]
            lmks[:,1] = self.rasterize_size-lmks[:,1]
            fit_params, _ = fitter.fit_orthographic_GN(lmks)
            R, _ = utils.get_rot_matrix(fit_params['u'])
            
            r = Rotation.from_matrix(R)
            # angles = r.as_mrp()
            angles = r.as_euler('xyz')
            # vec = angles.tolist()+[fit_params['taux'], fit_params['tauy'], fit_params['sigmax'], fit_params['sigmay']]
            # vec = np.array(vec)
            
            taus = fitter.to_projective(fit_params, cam.get_matrix(), lmks)
            if label_path.find('angle') > -1:
                vec = np.array(angles.tolist()+taus)
            else:
                vec = np.array(fit_params['u'].tolist()+taus)
            np.savetxt(label_path, vec)
            # break


mm = morphable_model.MorphableModel()


rdir = '/online_data/face/3dshape/for_3DID/cropped'
train_data = SimpleDataset(transform=Grayscale(num_output_channels=3), is_train=True, normalize_labels=False,
                           rootdir=rdir)
# train_data.create_labels(mm)

#%%
train_data = SimpleDataset(transform=Grayscale(num_output_channels=3), is_train=True, normalize_labels=False, rootdir=rdir)
test_data = SimpleDataset(transform=Grayscale(num_output_channels=3), is_train=False, normalize_labels=False, rootdir=rdir)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

device = 'cuda'

fov = 12.56
cx = 112
rasterize_size = 2*cx

# model = simple_model.SimpleModel()
model = medium_model.MediumModel(rasterize_fov=fov, rasterize_size=rasterize_size)


model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.scheduler_step_size)


# model.freeze_all_but_rigid_layers()
hist_tra = []
hist_tes = []

checkpoint_dir = 'checkpoints'

os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_file0 = f'{checkpoint_dir}/medium_modelss.pth'

if os.path.exists(checkpoint_file0):
    checkpoint = torch.load(checkpoint_file0)
    model.load_state_dict(checkpoint['model_state'])
    hist_tra = checkpoint['hist_tra']
    hist_tes = checkpoint['hist_tes']

#%%


### image level loss
def photo_loss(imageA, imageB, mask, eps=1e-6):
    """
    l2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
    Parameters:
        imageA       --torch.tensor (B, 3, H, W), range (0, 1), RGB order 
        imageB       --same as imageA
    """
    diff = imageA - imageB
    diff = diff**2
    loss = torch.sqrt(eps + torch.sum(diff, dim=1, keepdims=True))
    # print(loss.shape)
    # loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    loss = torch.sum(loss)/torch.max(torch.sum(mask), torch.tensor(1.0))
    return loss


def perceptual_loss(id_featureA, id_featureB):
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
    return torch.sum(1 - cosine_d) / cosine_d.shape[0]  


def reg_loss(params, sa, sb, l1=0.00005, l2=0.001, l3=1000.002):
    # sa: sigma alphas (st dev of shape)
    return l1*torch.linalg.norm(params['alpha']/sa) + l2*torch.linalg.norm(params['beta']/sb) + l3*torch.linalg.norm(params['exp'])/79.



def lm_loss(lmks, plmks, lmd1=0.0005):
    B = lmks.shape[0]
    return lmd1*torch.sum((lmks.reshape(B,-1)-plmks.reshape(B,-1))**2)

sa = model.mm.sigma_alphas
sb = model.mm.sigma_betas

# model.unfreeze_all_but_rigid_layers()
# model.freeze_all_but_rigid_layers()

for n in range(0, 1000):
    print(n)
    
    train_loss = 0
    model.train()
    num_batches = 0
    t0 = time()
    for train_features, train_labels, lmks, tforms in train_dataloader:
        # print(train_features.shape)

        inputs  = train_features.to(device)/255.
        targets = train_labels.to(device)
        lmks = lmks.to(device)[:,17:,:]
        tforms = tforms.to(device)
        
        if len(hist_tes) < 30:
            output, _, _ , _, _, _, _ = model(inputs, tforms=tforms, render=False)
            params = model.parse_params(output)
            closs = F.l1_loss(output[:,-6:], targets[:,:])
        else:
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
            
            model.freeze_rigid_layers()

            output, masked_in, rendered_out, mask, plmks, gt_feat, pred_feat = model(inputs, tforms=tforms, render=True)
            plmks[:,:,1] = 224-plmks[:,:,1]

            params = model.parse_params(output)
            # closs = F.l1_loss(output[:,-6:], targets[:,:]) + \
            #     photo_loss(masked_in, rendered_out, mask) + \
            #         reg_loss(params) + \
            #             lm_loss(lmks, plmks) + \
            #                 perceptual_loss(gt_feat, pred_feat)
                        
            closs = photo_loss(masked_in, rendered_out, mask) + \
                        reg_loss(params, sa, sb) + \
                            lm_loss(lmks, plmks) + \
                                perceptual_loss(gt_feat, pred_feat)
                        

        # _, _, ims = model.render_image(params)
        # closs = F.l1_loss(output[:,-6:], targets[:,:]) 
        # loss = F.l1_loss(output[:,:], targets[:,:3]) 
        train_loss += closs.item() #* inputs.size(0)

        optimizer.zero_grad()
        closs.backward()
        optimizer.step()
        num_batches += 1
        # params['tau'][:,-1] += 1000
        # (mask, depth, ims), lmks = model.render_image(params)
        # break
        # break
    # plt.clf()
    # plt.imshow(ims[0].detach().cpu().numpy()[0,:,:])
    # plt.plot(lmks[0][:,0], lmks[0][:,1], '.')
    # plt.show()
    # # break
      
    hist_tra.append(train_loss/num_batches)
    
    model.eval()
    
    with torch.no_grad():
        
        tes_loss = 0
        num_batches = 0
        for test_features, test_labels, lmks, tforms in test_dataloader:
            inputs  = test_features.to(device)/255.
            targets = test_labels.to(device)
            lmks = lmks.to(device)[:,17:,:]
            tforms = tforms.to(device)
            
            if len(hist_tes) < 30:
                output, _, _, _, plmks, _, _ = model(inputs, tforms, render=False)
                params = model.parse_params(output)
                closs = F.l1_loss(output[:,-6:], targets[:,:])
            else:
                output, masked_in, rendered_out, mask, plmks, gt_feat, pred_feat = model(inputs, tforms=tforms, render=True)
                plmks[:,:,1] = 224-plmks[:,:,1]
                params = model.parse_params(output)

                # closs = F.l1_loss(output[:,-6:], targets[:,:]) + \
                    # photo_loss(masked_in, rendered_out, mask) + \
                        # reg_loss(params) + \
                            # lm_loss(lmks, plmks) + \
                                # perceptual_loss(gt_feat, pred_feat)     
                closs = photo_loss(masked_in, rendered_out, mask) + \
                            reg_loss(params, sa, sb) + \
                                lm_loss(lmks, plmks) + \
                                    perceptual_loss(gt_feat, pred_feat)

                            
            tes_loss += closs.item() #* inputs.size(0)
            
            num_batches += 1
        
        hist_tes.append(tes_loss/num_batches)
        
        if n > 1 and (hist_tes[-1] < min(hist_tes[:-1])):
            checkpoint_file = f'{checkpoint_dir}/medium_modelsv_{n:05d}.pth'

            checkpoint = {
                "model_state": model.state_dict(),
                "hist_tra": hist_tra,
                "hist_tes": hist_tes,
                "nb_epochs_finished": n,
                "cuda_rng_state": torch.cuda.get_rng_state()
            }
            torch.save(checkpoint, checkpoint_file0)
            torch.save(checkpoint, checkpoint_file)
    
        # params['tau'][:,-1] += 1000
        
        # (_, _, ims), lmks = model.render_image(params)
        
        plt.clf()
        # plt.semilogy(hist_tra[25:])
        if plmks is not None:
            plt.semilogy(hist_tes[25:])
        else:
            plt.semilogy(hist_tes)

        plt.show()
        
        
        if plmks is not None:
            plt.clf()
            plt.figure(figsize=(14,14))
            plt.subplot(121)
            plt.imshow(masked_in[0].detach().cpu().numpy()[0,:,:])
            # plt.plot(lmks[0][:,0].cpu(), lmks[0][:,1].cpu(), '.')
            # plt.plot(plmks[0][:,0].detach().cpu(), plmks[0][:,1].detach().cpu(), '.')
    
            # lmks[:,:,1] = 224-lmks[:,1]
    
            # plt.plot(lmks[0][:,0], lmks[0][:,1], '.')
            plt.subplot(122)
            plt.imshow(rendered_out[0,0,:,:].detach().cpu().numpy())
            # plt.plot(lmks[0][:,0], lmks[0][:,1], '.')

        plt.show()
    
    print('%.2f seconds' % (time()-t0))

#%%
ims = train_features
cropped = resize_n_crop(ims, tforms)
plt.imshow(ims[0,0,:,:].detach().cpu())
plt.imshow(cropped[0,0,:,:].detach().cpu())

