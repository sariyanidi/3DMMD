#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:53:57 2023

@author: v
"""

import sys
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from models.morphable_model import MorphableModel
from models.camera import Camera
from models.orthograhic_fitter import OrthographicFitter
from time import time
# 

sys.path.append('../')

from utils import utils
from glob import glob

import matplotlib.pyplot as plt


class PerspectiveFitter():
    
    def __init__(self, mm, cam, F=7, use_ineq=False, use_maha=True, which_pts='all', 
                 maha2_threshold=256.0,
                 which_basis='identity'):
        self.mm = mm
        self.F = F
        self.which_pts = which_pts
        self.maha2_threshold = maha2_threshold
        self.which_basis = which_basis
        if self.which_basis == 'identity':
            self.num_components = self.mm.Kid
        elif self.which_basis == 'expression':
            self.num_components = self.mm.Kexp
            
        NALLPTS = len(self.mm.mean_shape[::3])

        if self.which_pts == 'lmks':
            self.pts_indices = self.mm.li
        elif self.which_pts == 'all':
            self.pts_indices = torch.range(0, NALLPTS-1).long().to(self.mm.device)
        elif self.which_pts == 'sampled':
            NALLPTS = len(self.mm.mean_shape[::3])
            sampled = torch.range(0, NALLPTS-1, 10).long().to(self.mm.device)
            self.pts_indices = torch.cat([self.mm.li, sampled])
        
        x0 = self.mm.mean_shape[::3][self.pts_indices].reshape(-1,1)
        y0 = self.mm.mean_shape[1::3][self.pts_indices].reshape(-1,1)
        z0 = self.mm.mean_shape[2::3][self.pts_indices].reshape(-1,1)
    
        self.NPTS = len(x0)
        
        self.cam = cam

        self.use_ineq = use_ineq
        self.use_maha = use_maha
        
        
        if self.which_basis == 'identity':
            self.alpha_ub = 6*self.mm.sigma_alphas.reshape(1,-1)
            self.alpha_lb = -6*self.mm.sigma_alphas.reshape(1,-1)
        elif self.which_basis == 'expression':
            self.alpha_ub = self.mm.eps_upper.reshape(1,-1)
            self.alpha_lb = self.mm.eps_lower.reshape(1,-1)
    
    
    def parse_vars_idx(self, f):
        alpha_i = 0
        taux_i = self.num_components+f*6
        tauy_i = self.num_components+f*6+1
        tauz_i = self.num_components+f*6+2
        u_i = self.num_components+f*6+3
        
        return (alpha_i, taux_i, tauy_i, tauz_i, u_i)
    
    
    def parse_vars(self, x, f):
        (alpha_i, taux_i, tauy_i, tauz_i, u_i) = self.parse_vars_idx(f)
        alpha = x[alpha_i:alpha_i+self.num_components]
        taux = x[taux_i]
        tauy = x[tauy_i]
        tauz = x[tauz_i]
        u = x[u_i:u_i+3]
        
        return (alpha, taux, tauy, tauz, u)
    
    
    def compute_gradient_and_hessian(self, x, all_lmks):
        
        K = len(x)
        obj = 0
        H = torch.zeros((K, K)).to(self.mm.device)
        dg = torch.zeros((1, K)).to(self.mm.device)
        
        xprojs = []
        yprojs = []
        
        for f in range(self.F):
            dg_f, H_f, xproj, yproj, obj_f = self.compute_gradient_and_hessian_f(x, all_lmks[f], f)
            dg += dg_f
            H += H_f
            obj += obj_f
            xprojs.append(xproj)
            yprojs.append(yproj)
        
        return dg, H, xprojs, yprojs, obj
            
    
    
    def compute_gradient_and_hessian_f(self, x, lmks, f):
        (alpha_i, taux_i, tauy_i, tauz_i, u_i) = self.parse_vars_idx(f)
        (alpha, taux, tauy, tauz, u) = self.parse_vars(x,f)
    
        R, dR_du = utils.get_rot_matrix_torch(u, self.mm.device)
        R = R.float()
        dR_du = dR_du.float()
        
        alpha = alpha.unsqueeze(0)
        if self.which_basis == 'identity':
            p = self.mm.compute_face_shape(alpha, None).squeeze().T
        elif self.which_basis == 'expression':
            # exp = alpha.unsqueeze(0)
            alpha_ = torch.zeros((1, self.mm.Kid)).to(self.mm.device)    
            p = self.mm.compute_face_shape(alpha_, alpha).squeeze().T
        
        p = p[:,self.pts_indices]

        
        # print(p.shape)
        Rp = R @ p
        
        v = Rp+torch.tensor([taux, tauy, tauz]).reshape(3,1).to(self.mm.device)
        vx = v[0:1,:].T
        vy = v[1:2,:].T
        vz = v[2:3,:].T
        
        inv_vz = 1./vz
        inv_vz2 = (inv_vz)**2
        
        if self.which_basis == 'identity':
            IR = self.mm.I.reshape(-1, 3, self.num_components).permute(2,0,1)
        elif self.which_basis == 'expression':
            IR = self.mm.E.reshape(-1, 3, self.num_components).permute(2,0,1)
        IR = IR[:,self.pts_indices,:]
        
        # print(IR.shape)
        
        dv_dalpha = (IR @ R.T).permute(1,2,0).reshape(-1, self.num_components)
        dvx_dalpha = dv_dalpha[0::3,:]
        dvy_dalpha = dv_dalpha[1::3,:]
        dvz_dalpha = dv_dalpha[2::3,:]
        
        dv_du1 = (dR_du[:, :, 0] @ p).T
        dv_du2 = (dR_du[:, :, 1] @ p).T
        dv_du3 = (dR_du[:, :, 2] @ p).T
        
        nablaWx_alpha = self.cam.f_x*inv_vz2*(dvx_dalpha * vz - dvz_dalpha*vx);
        nablaWy_alpha = self.cam.f_y*inv_vz2*(dvy_dalpha * vz - dvz_dalpha*vy);
        
        nablaWx = torch.zeros(self.NPTS,len(x)).to(self.mm.device)
        nablaWx[:,alpha_i:alpha_i+self.num_components] = nablaWx_alpha
        nablaWx[:,taux_i:taux_i+1] = self.cam.f_x*inv_vz
        nablaWx[:,tauz_i:tauz_i+1] = -self.cam.f_x*vx*inv_vz2
        nablaWx[:,u_i:u_i+1] = (self.cam.f_x*inv_vz2)*(vz*dv_du1[:,0:1]-vx*dv_du1[:,2:3])
        nablaWx[:,u_i+1:u_i+2] = (self.cam.f_x*inv_vz2)*(vz*dv_du2[:,0:1]-vx*dv_du2[:,2:3])
        nablaWx[:,u_i+2:u_i+3] = (self.cam.f_x*inv_vz2)*(vz*dv_du3[:,0:1]-vx*dv_du3[:,2:3])
        
        nablaWy = torch.zeros(self.NPTS,len(x)).to(self.mm.device)
        nablaWy[:,alpha_i:alpha_i+self.num_components] = nablaWy_alpha
        nablaWy[:,tauy_i:tauy_i+1] = self.cam.f_y*inv_vz
        nablaWy[:,tauz_i:tauz_i+1] = -self.cam.f_y*vy*inv_vz2
        nablaWy[:,u_i:u_i+1] = (self.cam.f_y*inv_vz2)*(vz*dv_du1[:,1:2]-vy*dv_du1[:,2:3])
        nablaWy[:,u_i+1:u_i+2] = (self.cam.f_y*inv_vz2)*(vz*dv_du2[:,1:2]-vy*dv_du2[:,2:3])
        nablaWy[:,u_i+2:u_i+3] = (self.cam.f_y*inv_vz2)*(vz*dv_du3[:,1:2]-vy*dv_du3[:,2:3])

        xproj = self.cam.f_x*vx/vz+self.cam.cx
        yproj = self.cam.f_y*vy/vz+self.cam.cy
        
        diffx = (xproj - lmks[self.pts_indices,0:1])
        diffy = (yproj - lmks[self.pts_indices,1:2])
        
        
        # dv_du1 = (dR_du(:,:,1)*p)';
        # dv_du2 = (dR_du(:,:,2)*p)';
        # dv_du3 = (dR_du(:,:,3)*p)';
        gradient = diffx.T @ nablaWx + diffy.T @ nablaWy
        nabla2F = nablaWx.T @ nablaWx + nablaWy.T @ nablaWy
        obj = torch.sum(diffx**2) + torch.sum(diffy**2)
        
        hessian = nabla2F
        
        if self.use_ineq and f == 0:
            # print(alpha.shape)
            # print(self.alpha_ub.shape)
            
            f_alpha_ub = alpha-self.alpha_ub
            f_alpha_lb = self.alpha_lb-alpha
            
            if self.use_maha:
                cmaha = (alpha*((self.mm.sigma_alphas)**(-2))*alpha).sum()
                f_maha = cmaha-self.maha2_threshold
                # print(f'{f_maha:.3f} == ')
                nabla_f_maha = -(1.0/f_maha)*2*alpha*((self.mm.sigma_alphas)**(-2))
                nabla_f_maha = nabla_f_maha.reshape(-1,1)
            
            nablaf_alpha_ub =  torch.diag(-1.0/f_alpha_ub.flatten())
            nablaf_alpha_lb = -torch.diag(-1.0/f_alpha_lb.flatten())
            # print(f_alpha_ub.shape)
            # print(nablaf_alpha_ub.shape)
            # print(nablaf_alpha_lb.shape)

            
            nablaf_alpha = torch.sum(nablaf_alpha_ub+nablaf_alpha_lb,axis=0)
            
            nabla2_phi_alpha = nablaf_alpha_ub @ nablaf_alpha_ub.T + nablaf_alpha_lb @ nablaf_alpha_lb.T
    
            hessian  *= self.opt_t
            gradient *= self.opt_t
            
            hessian[:self.num_components, :self.num_components] += nabla2_phi_alpha 
            gradient[0,:self.num_components] += nablaf_alpha
            
            if self.use_maha:
                hessian[:self.num_components, :self.num_components] += nabla_f_maha @ nabla_f_maha.T
                gradient[0,:self.num_components] += nabla_f_maha.flatten()
            
            obj *= self.opt_t
        
            obj += - torch.sum(torch.log(-f_alpha_lb)) - torch.sum(torch.log(-f_alpha_ub))
            if self.use_maha:
                obj += - torch.log(-f_maha)

        return gradient, hessian, xproj, yproj, obj
        
    
    
    def evaluate_function(self, x, all_lmks):
        obj = 0
        RMSE = 0
        
        for f in range(self.F):
            obj_f, RMSE_f = self.evaluate_function_f(x, all_lmks[f], f)
            obj += obj_f
            RMSE += RMSE_f
            
        return obj, RMSE
    
    
    def construct_initialization(self, alphas, taus, us):
        alpha = torch.stack(alphas).mean(dim=0)
        poses = []
        for i in range(len(taus)):
            poses.append(torch.cat([taus[i], us[i]]))
        
        return torch.cat([alpha, torch.cat(poses)])
            

    

    def evaluate_function_f(self, x, lmks, f):
        (alpha, taux, tauy, tauz, u) = self.parse_vars(x, f)

        R, _ = utils.get_rot_matrix_torch(u, self.mm.device)
        R = R.float()
                
        alpha = alpha.unsqueeze(0)

        if self.which_basis == 'identity':
            p = self.mm.compute_face_shape(alpha, None).squeeze().T
        elif self.which_basis == 'expression':
            # exp = alpha.unsqueeze(0)
            alpha_ = torch.zeros((1, self.mm.Kid)).to(self.mm.device)    
            p = self.mm.compute_face_shape(alpha_, alpha).squeeze().T
        
        
        p = p[:,self.pts_indices]

        # print(p.shape)
        Rp = R @ p

        
        v = Rp+torch.tensor([taux, tauy, tauz]).reshape(3,1).to(self.mm.device)
        vx = v[0:1,:].T
        vy = v[1:2,:].T
        vz = v[2:3,:].T
        
        xproj = self.cam.f_x*vx/vz+self.cam.cx
        yproj = self.cam.f_y*vy/vz+self.cam.cy
        
        diffx = (xproj - lmks[self.pts_indices,0:1])
        diffy = (yproj - lmks[self.pts_indices,1:2])
        

        if self.use_maha:
            cmaha = (alpha*((self.mm.sigma_alphas)**(-2))*alpha).sum()
            f_maha = cmaha-self.maha2_threshold
        
        obj = torch.sum(diffx**2) + torch.sum(diffy**2)
        
        if self.use_ineq:
            obj *= self.opt_t    
            f_alpha_ub = alpha-self.alpha_ub
            f_alpha_lb = self.alpha_lb-alpha

            obj += - torch.sum(torch.log(-f_alpha_lb)) - torch.sum(torch.log(-f_alpha_ub))
            if self.use_maha:
                obj += -torch.log(-f_maha)
    
        RMSE = torch.sqrt(obj) / len(diffx)
    
        return obj, RMSE

    
        
    def fit_GN(self, x0, all_lmks, use_ineq=False, plotit=False, plotlast=False):
        self.use_ineq = use_ineq
        
        ALPHA = 0.2
        BETA = 0.75
    
        MAXITER = 300*self.F
        dlt = 0.001
        if self.use_ineq == True:
            MAXITER = 500*self.F
            dlt = 0.0
            
        objs = torch.zeros(MAXITER)
        x = x0
        
        self.opt_t = 1
    
        for i in range(MAXITER):
            # print(i)
            
            if i % 10 == 0:
                if self.use_ineq:
                    self.opt_t *= 5
                # dlt /= 10
                
            dg, H, xprojs, yprojs, obj = self.compute_gradient_and_hessian(x, all_lmks)
            dg = dg.T
            
            H[:self.num_components,:self.num_components] += dlt*torch.eye(self.num_components).to(self.mm.device)
            search_dir = -torch.linalg.solve(H, dg)
            # print(search_dir)
            if plotit:
                plt.clf()
                plt.figure(figsize=(20,20))
                plt.plot(all_lmks[0][:,0].cpu(), all_lmks[0][:,1].cpu(), 'o')
                plt.plot(xprojs[0].cpu(), yprojs[0].cpu(), 'x')
                # plt.xlim((0, 1440))
                # plt.ylim((0, 1920))
                plt.show()
                # plt.plot(x[:self.num_components])
                # plt.show()
                print('-'*20)
                print(obj)
                print('='*20)

    
            maxiter = 100
            it = 0
            t = 1

            terminate = False
            while it < maxiter:
                xtmp = x + t * search_dir.reshape(-1,)
                obj_tmp, _ = self.evaluate_function(xtmp.float(), all_lmks)
                
                # if torch.isnan(obj_tmp)d and it >= 1:
                    # search_dir[:self.num_components] *= 0
                    # obj_tmp, _ = self.evaluate_function(xtmp.float(), all_lmks)
                # print(obj)
                # print(obj_tmp)
                
                if obj_tmp < obj + ALPHA * t * (dg.T @ search_dir):
                    break
                
                if t < 1e-4:
                    terminate = True
                    break
                
                t = BETA * t
                it += 1
            
            objs[i] = obj
            # print(t)
            
            if torch.linalg.norm(search_dir) < 1e-7:
                break
            
            if terminate:
                break
    
            x = x + t * search_dir.reshape(-1,)
            x = x.float()
    

        if self.use_ineq and plotlast:
            for f in range(self.F):
                plt.figure(figsize=(60,40))
                plt.subplot(231)
                plt.plot(all_lmks[f][:,0].cpu(), all_lmks[f][:,1].cpu(), 'o')
                plt.plot(xprojs[f].cpu(), yprojs[f].cpu(), 'x')
                plt.subplot(232)
                plt.plot(all_lmks[f][:,0].cpu(), all_lmks[f][:,1].cpu(), 'o')
                plt.subplot(233)
                plt.plot(xprojs[f].cpu(), yprojs[f].cpu(), 'x')
                
                if self.which_basis == 'expression':
                    alpha_ = torch.zeros(1,self.mm.Kid).to(self.mm.device)
                    eps_ = x[:self.mm.Kexp].reshape(1,self.mm.Kexp)
                elif self.which_basis == 'identity':
                    alpha_  = x[:self.mm.Kid].reshape(1,self.mm.Kid)
                    eps_ = torch.zeros(1,self.mm.Kexp).to(self.mm.device)
                
                
                p0 = self.mm.compute_face_shape(alpha_, eps_).squeeze()
                plt.subplot(234)
                plt.plot(p0[:,0].cpu(), p0[:,1].cpu(), 'x')

                plt.show()
                # plt.plot(x[:self.num_components])
                # plt.show()
    


        objs = objs[:i]
        # plt.figure()
        # plt.plot(objs)
        
        fit_params = {
            'taux': x[0],
            'tauy': x[1],
            'sigmax': x[2],
            'sigmay': x[3],
            'u': x[4:],
            'num_iters': i,
            'xproj': xprojs,
            'yproj': yprojs,
        }
    
        return x, fit_params
    
#%%
import scipy.io

from scipy.spatial import distance
import os



shp = scipy.io.loadmat('/home/sariyanide/car-vision/matlab/geometric_error/data/01_MorphableModel.mat')['shapeMU'].reshape(-1,3)
N = shp.shape[0]
# A copy 
bdir = '/home/sariyanide/car-vision/python/geometric_error_analysis/'
w = scipy.io.loadmat(f'{bdir}/idxs/BFM_exp_idx.mat')['trimIndex'].astype(int)
w0 = (w-1).flatten()

        
# Deep3DFace
ix = scipy.io.loadmat(f'{bdir}/idxs/Deep3DFace/BFM_front_idx.mat')['idx'].astype(int)
ix0 = (ix-1).flatten()
ix_d = w0[ix0]

# 3DDFA
ix = np.loadtxt(f'{bdir}/idxs/3DDFA/indices.txt').astype(int)
ix_3ddfa = w0[ix]

# 3DI
ix = np.loadtxt(f'{bdir}/idxs/3DI/w.dat').astype(int)
ix_3di = w0[((ix-1)[::3]/3).astype(int)]

# 3DI
ix = np.loadtxt(f'{bdir}/idxs/3DI/w.dat').astype(int)
ix_3di = w0[((ix-1)[::3]/3).astype(int)]

# The points that are common across methods
common_idx = set(ix_d).intersection(set(ix_3ddfa)).intersection(set(ix_3di))
common_idx = list(common_idx)

which = '3'
which_pts = 'sampled'

if __name__ == '__main__':
    mm = MorphableModel(key='BFMmm-23660', data_rootdir='../data')
    of = OrthographicFitter(mm)
    # sdir = '/offline_data/face/yt_faces2/3DI/OD0.6_OE0.3_OB0.7_RS78_IN0_SF0_CF0.42NF7_NTC7_UL0_CB1GLOBAL79_IS0K7440.75P1BFMmm-19830/10/'
    # files = glob(f'{sdir}/*pts')
    sdir = f'/home/sariyanide/code/3DMMD/tmp{which}'

    F = 9
    avg_mahas = []
    
    
    for fov in [12]:
        cam = Camera(fov, fov, 112, 112)
        
        pfs_f = PerspectiveFitter(mm, cam, use_maha=False, which_pts='lmks', F=1)
        pfd_f = PerspectiveFitter(mm, cam, use_maha=False, which_pts=which_pts, F=1)
        
        pfs = PerspectiveFitter(mm, cam, use_maha=True, which_pts='lmks', F=F, maha2_threshold=300)
        pfd = PerspectiveFitter(mm, cam, use_maha=True, which_pts=which_pts, F=F, maha2_threshold=300)
        # fov = 17
        for subj_id in range(33):# range(100):
        # for fov in [1, 2, 5, 10, 17, 20, 25, 30, 50]:
        # for fov in [1, 2, 4, 7, 12]:
        # for fov in [1, 5, 10, 17]:
            tdir = f'/offline_data/face/meshes/synth_020/3DID-{which}fov{fov}-{which_pts}-M01-F{F:02d}'
            os.makedirs(tdir, exist_ok=True)
            files = glob(f'{sdir}/subj{subj_id:03d}*txt')
            
            tpath = f'{tdir}/subj{subj_id:03d}.txt'
            if os.path.exists(tpath):
                continue
            
            plotit = False
            
            alphas = []
            taus_us = []
            for fpath in files[:F]:
                # print(os.path.basename(fpath))
                # plt.clf()
                
                
                cpts = np.loadtxt(fpath)
                all_pts = [cpts]
                all_lmks = [all_pts[0][pfs.mm.li.cpu().numpy(),:]]
                of_fit_params = of.fit_orthographic_GN(all_lmks[0], plotit=plotit)[0]
                u = of_fit_params['u']
                tau = of.to_projective(of_fit_params, cam.get_matrix(), all_lmks[0])

                x0 = torch.cat([0*torch.rand(pfd.num_components), torch.tensor(tau), torch.tensor([0.1, 0.2, 0.3])])
                # x0 = torch.cat([0*torch.rand(199), torch.tensor(tau), torch.tensor(u)])
                # x0 = torch.cat([0*torch.rand(199), torch.tensor([0.1, 0.2, 599]), torch.tensor([0.1, 0.2, 0.3])])

        
                x0 = pfs_f.fit_GN(x0.float(), all_pts, plotit=plotit, use_ineq = False)[0]
                # print(tau)
                # print(x0[199:])
                
                x0[:pfs.mm.Kid] *= 0
                x0 = pfs_f.fit_GN(x0.float(), all_pts, plotit=plotit, use_ineq = True)[0]
                x = pfd_f.fit_GN(x0.float(), all_pts, plotit=plotit, use_ineq = True)[0]
                
                alpha = x[:pfd.mm.Kid]
                tau = x[pfd.mm.Kid:pfd.mm.Kid+3]
                u =  x[pfd.mm.Kid+3:pfd.mm.Kid+6]
                
                alphas.append(alpha)
                taus_us.append(torch.cat([tau, u]))
                # break
            # break
            alpha =torch.stack(alphas).mean(axis=0)
           
            all_pts = []
            all_lmks = []
            
            mahas = []
             
            
            plotit2 = False
            x0 = torch.cat([alpha] + taus_us)
            # x_list = [0*torch.rand(199)]
            for fpath in files:
                # print(os.path.basename(fpath))
                # plt.clf()
                pts = np.loadtxt(fpath)
                lmks = pts[pfs.mm.li.cpu().numpy(),:]
                lmks = pts[pfs.mm.li.cpu().numpy(),:]
                all_pts.append(pts)
                all_lmks.append(lmks)
                # lmks[:,1] = 224-lmks[:,1]
                
                # x_list.append(torch.tensor([0.1, 0.2, 599]))
                # x_list.append(torch.tensor([0.1, 0.2, 0.3]))
                
        
            # x0 = torch.cat(x_list)
            # x0 = pfs.fit_GN(x0.float(), all_lmks, plotit=False, use_ineq = False)[0]
            # x0[:pfs.mm.Kid] = alpha
            # x0 = pfs.fit_GN(x0.float(), all_lmks, plotit=False, use_ineq = True)[0]
            x = pfd.fit_GN(x0.float(), all_pts, plotit=plotit2, use_ineq = True)[0]
            # x = pfs.fit_GN(x0.float(), all_pts, plotit=plotit2, use_ineq = True)[0]
            # x = pfd.fit_GN(x.float(), pts, plotit=True, use_ineq = False)[0]
            
            
            alpha = x[:pfd.mm.Kid]
            VI = np.diag((pfd.mm.sigma_alphas.cpu().numpy())**(-2))
            d_maha = distance.mahalanobis(alpha, 0*alpha, VI)
            mahas.append(d_maha)
            print(d_maha)
            # plt.plot(alpha)
            # break
            R = pfd.mm.compute_face_shape(alpha.unsqueeze(0).to('cuda')).cpu().squeeze().numpy()
            # break

            w = np.loadtxt('%s/car-vision/python/geometric_error/ridxs/ix_3di.txt' % os.path.expanduser('~')).astype(int)
            # R = pfd.mm.compute_face_shape(torch.from_numpy(alpha).to('cuda'))
            sdir = '/home/sariyanide/code/3DMMD/tmp'
            """
            Rpaths = glob(f'{sdir}/R{subj_id:03d}*txt')
            Rs = []
            for Rpath in Rpaths:
                Rs.append(np.loadtxt(Rpath))
            R = np.mean(np.array(Rs), axis=0)
            # R = R[w]
            """

           
            R_full = np.zeros((N,3))
    
            R_full[ix_3di,:] = R # np.loadtxt(files[0])
            R = R_full[common_idx,:]
            np.savetxt(tpath, R)
            
            G = np.loadtxt(f'/offline_data/face/meshes/synth_020/ground_truth/subj{subj_id:03d}.txt')
            plt.figure(figsize=(40, 20))
            plt.subplot(121)
            plt.plot(G[:,0], -G[:,1], '.')
            plt.subplot(122)
            plt.plot(G[:,2], -G[:,1], '.')
            plt.show()
            plt.figure(figsize=(40, 20))
            plt.subplot(121)
            plt.plot(R[:,0], R[:,1], '.')
            plt.subplot(122)
            plt.plot(R[:,2], R[:,1], '.')
            plt.show()
            """

            liBFM = [19106,19413,19656,19814,19981,20671,20837,20995,21256,21516,8161,8175,8184,8190,6758,7602,8201,8802,9641,1831,3759,5049,6086,4545,3515,10455,11482,12643,14583,12915,11881,5522,6154,7375,8215,9295,10523,10923,9917,9075,8235,7395,6548,5908,7264,8224,9184,10665,8948,8228,7508]
            lrigid = np.array(liBFM)[[13, 19, 28, 31, 37]]
            
            _, __, tform = procrustes(R[lrigid], G[lrigid])
            
            G = tform['scale']*(G @ tform['rotation'])+tform['translation']
             
        
        
            
            err = compute_error(R, G)
            print('='*89)
            print(f'{fov}: {np.mean(err)}')
            """
        
        

#%%
        
"""
if __name__ == '__main__':
    of = OrthographicFitter(mm)
    # sdir = '/offline_data/face/yt_faces2/3DI/OD0.6_OE0.3_OB0.7_RS78_IN0_SF0_CF0.42NF7_NTC7_UL0_CB1GLOBAL79_IS0K7440.75P1BFMmm-19830/10/'
    # files = glob(f'{sdir}/*pts')
    sdir = '/home/sariyanide/code/3DMMD/tmp'
    files = glob(f'{sdir}/*txt')
    
    
    avg_mahas = []
    fovs = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60]
    fovs = [20]
    for fov in fovs:
        print(fov)
        cam = Camera(fov, fov, 112, 112)
        pfs = PerspectiveFitter(mm, cam, use_maha=True, dense=False)
        pfd = PerspectiveFitter(mm, cam, use_maha=True, dense=True)
    
        mahas = []
        for fpath in files:
            # print(os.path.basename(fpath))
            # plt.clf()
            pts = np.loadtxt(fpath)
            lmks = pts[pfs.mm.li.cpu().numpy(),:]
            # lmks[:,1] = 224-lmks[:,1]
    
            x0 = torch.cat([0*torch.rand(199), torch.tensor([0.1, 0.2, 599]), torch.tensor([0.1, 0.2, 0.3])])
            x0 = pfs.fit_GN(x0.float(), lmks, plotit=False, use_ineq = False)[0]
            x0[:pfs.mm.Kid] *= 0
            x0 = pfs.fit_GN(x0.float(), lmks, plotit=False, use_ineq = True)[0]
            x = pfd.fit_GN(x0.float(), pts, plotit=False, use_ineq = True)[0]
            # x = pfd.fit_GN(x.float(), pts, plotit=True, use_ineq = False)[0]
            
            
            alpha = x[:pfd.mm.Kid]
            VI = np.diag((pfd.mm.sigma_alphas.cpu().numpy())**(-2))
            d_maha = distance.mahalanobis(alpha, 0*alpha, VI)
            mahas.append(d_maha)
            # print(d_maha)
            # plt.plot(alpha)
            # break
        avg_mahas.append(np.mean(mahas))
    #%%
    
    w = np.loadtxt('%s/car-vision/python/geometric_error/ridxs/ix_3di.txt' % os.path.expanduser('~')).astype(int)
    plt.plot(fovs, avg_mahas)
    # R = pfd.mm.compute_face_shape(torch.from_numpy(alpha).to('cuda'))
    R = pfd.mm.compute_face_shape(alpha.unsqueeze(0).to('cuda')).cpu().squeeze().numpy()
    
    #%%
    G = np.loadtxt('/offline_data/face/meshes/synth_020/ground_truth/subj003.txt')
    plt.figure(figsize=(40, 20))
    plt.subplot(121)
    plt.plot(G[:,0], -G[:,1], '.')
    plt.subplot(122)
    plt.plot(G[:,2], -G[:,1], '.')
    
    plt.figure(figsize=(40, 20))
    plt.subplot(121)
    plt.plot(R[:,0], R[:,1], '.')
    plt.subplot(122)
    plt.plot(R[:,2], R[:,1], '.')
    
        """
