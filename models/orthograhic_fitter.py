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

sys.path.append('../')

from utils import utils
from glob import glob

import matplotlib.pyplot as plt

class OrthographicFitter():
    
    
    def __init__(self, mm):
        self.mm = mm
        x0 = self.mm.mean_shape[::3][self.mm.li].reshape(-1,1).cpu().numpy()
        y0 = self.mm.mean_shape[1::3][self.mm.li].reshape(-1,1).cpu().numpy()
        z0 = self.mm.mean_shape[2::3][self.mm.li].reshape(-1,1).cpu().numpy()
        
        self.p0L = np.concatenate((x0, y0, z0), axis=1).T
        self.p0L -= self.p0L.mean(axis=1).reshape(-1,1)
        # self.p0L /= np.std(self.p0L)
        
    
    def compute_gradient_and_hessian(self, x, lmks):
        taux = x[0]
        tauy = x[1]
        sigmax = x[2]
        sigmay = x[3]
        u = x[4:]
        # print(u)
    
        L = len(self.mm.li)
    
        R, dR_du = utils.get_rot_matrix(u)
    
        Rp = R @ self.p0L
        r1p = Rp[0, :]
        r2p = Rp[1, :]
        
    
        dRp_du1 = dR_du[:, :, 0] @ self.p0L
        dRp_du2 = dR_du[:, :, 1] @ self.p0L
        dRp_du3 = dR_du[:, :, 2] @ self.p0L
    
        dr1p_du1 = dRp_du1[0, :].T
        dr2p_du1 = dRp_du1[1, :].T
        dr1p_du2 = dRp_du2[0, :].T
        dr2p_du2 = dRp_du2[1, :].T
        dr1p_du3 = dRp_du3[0, :].T
        dr2p_du3 = dRp_du3[1, :].T
    
        nablaWx = np.column_stack((sigmax * np.ones(L), np.zeros(L), taux + r1p, np.zeros(L), sigmax * dr1p_du1, sigmax * dr1p_du2, sigmax * dr1p_du3))
        nablaWy = np.column_stack((np.zeros(L), sigmay * np.ones(L), np.zeros(L), tauy + r2p, sigmay * dr2p_du1, sigmay * dr2p_du2, sigmay * dr2p_du3))
    
        nablaW = np.vstack((nablaWx, nablaWy))
        # print(nablaW.shape)
    
        hessian = np.dot(nablaW.T, nablaW)
        # plt.imshow(hessian[2:,2:])
        # print(np.linalg.matrix_rank(hessian[4:,4:]))
        # print(hessian[4:,4:])
        xproj = sigmax * (r1p + taux)
        yproj = sigmay * (r2p + tauy)
    
        diffx = (xproj - lmks[:,0])
        diffy = (yproj - lmks[:,1])
            
        err = np.hstack((diffx, diffy))
    
        gradient = np.dot(nablaW.T, err)
    
        obj = np.sum(diffx**2) + np.sum(diffy**2)
    
        return gradient, hessian, xproj, yproj, obj
        
    
    def evaluate_function(self, x, lmks):
        taux = x[0]
        tauy = x[1]
        sigmax = x[2]
        sigmay = x[3]
        u = x[4:]
    
        R, _ = utils.get_rot_matrix(u)
    
        Rp = R @ self.p0L
        r1p = Rp[0, :]
        r2p = Rp[1, :]
    
        xproj = sigmax * (r1p + taux)
        yproj = sigmay * (r2p + tauy)
    
        diffx = (xproj - lmks[:,0])
        diffy = (yproj - lmks[:,1])
    
        obj = np.sum(diffx**2) + np.sum(diffy**2)
    
        RMSE = np.sqrt(obj) / len(diffx)
    
        return obj, RMSE

    
        
    def fit_orthographic_GN(self, lmks, taux0=0.004, tauy0=0.002, sigma0=1, u0=None, plotit=False):
        import copy
        if u0 is None:
            u0 = np.array([1.1, 0.2, 0.3])
    
        taux = taux0
        tauy = tauy0
        sigmax = sigma0
        sigmay = sigma0
    
        xmean = np.mean(lmks[:,0])
        ymean = np.mean(lmks[:,1])
        
        lmks_0mean = copy.deepcopy(lmks)
        # print(lmks)
        
        lmks_0mean[:,0] = lmks[:,0] - xmean
        lmks_0mean[:,1] = lmks[:,1] - ymean
    
        ALPHA = 0.4
        BETA = 0.5
    
        MAXITER = 30
        objs = np.zeros(MAXITER)
        x = np.array([taux, tauy, sigmax, sigmay] + list(u0))
        
        for i in range(MAXITER):
            # print(i)
            dg, H, xproj, yproj, obj = self.compute_gradient_and_hessian(x, lmks_0mean)

            dg = dg.T
            search_dir = -np.linalg.solve(H, dg)
    
            if plotit:
                plt.clf()
                plt.plot(xproj, -yproj)
                plt.plot(lmks_0mean[:,0], -lmks_0mean[:,1])
                plt.show()
    
            maxiter = 1000
            it = 0
            t = 1
    
            while it < maxiter:
                xtmp = x + t * search_dir
    
                obj_tmp, _ = self.evaluate_function(xtmp, lmks_0mean)
    
                if obj_tmp < obj + ALPHA * t * np.dot(dg, search_dir):
                    break
                
                if t < 1e-7:
                    break
                
                t = BETA * t
                it += 1
    
            objs[i] = obj
    
            if np.linalg.norm(search_dir) < 1e-6:
                break
    
            x = x + t * search_dir
    
        objs = objs[:i]
        # plt.figure()
        # plt.plot(objs)
    
        _, RMSE = self.evaluate_function(x, lmks_0mean)
    
        fit_params = {
            'taux': x[0],
            'tauy': x[1],
            'sigmax': x[2],
            'sigmay': x[3],
            'u': x[4:],
            'num_iters': i,
            'RMSE': RMSE,
            'xproj': xproj,
            'yproj': yproj,
        }
    
        return fit_params, RMSE
    
    
    def to_projective(self, fit_params, camera, lmks):
        phix = camera[0,0]
        phiy = camera[1,1]
        cx = camera[0,2]
        cy = camera[1,2]
        
        
        sigmax = fit_params['sigmax']
        sigmay = fit_params['sigmay']
        R, _ = utils.get_rot_matrix(fit_params['u'])
        tauz = phix/((np.abs(sigmax)+np.abs(sigmay))*0.5)
        
        
        meanx = lmks[:,0].mean()
        meany = lmks[:,1].mean()
        
        # taux[0] = tauz[0]*(meanx[0]-cx[0])/phix[0];
        # tauy[0] = tauz[0]*(meany[0]-cy[0])/phiy[0];
        
        pmx = np.mean(self.p0L[:,0])
        pmy = np.mean(self.p0L[:,1])
        
        taux = tauz*((meanx)-cx)/phix
        tauy = tauz*((meany)-cy)/phiy
        
        
        # print(R.flatten().tolist())
        # print(taux)
        # print(tauy)
        # print(tauz)
        
        # print(meanx)
        # print(lmks)

        # print(taux)
        # []
        # tauz = 540
        
        V = R @ self.p0L + np.array([taux, tauy, tauz]).reshape(-1,1)
        # print(self.p0L)
        # print(V)
        # print(fit_params)
        x = phix*V[0,:]/V[2,:]+cx
        y = phiy*V[1,:]/V[2,:]+cy
        
        # plt.plot(lmks[:,0], lmks[:,1], '.')
        # plt.plot(x, y, '.')
        # plt.xlim((0, 2*cx))
        # plt.ylim((0, 2*cy))
        # ax = plt.gca()
        # ax.set_aspect('equal', adjustable='box')
        # plt.draw()
        
        return [taux, tauy, tauz]

