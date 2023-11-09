#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:13:11 2023

@author: v
"""
import numpy as np
from utils import utils
import torch
import torch.nn.functional as F

def compute_rotation_matrix_from_eulerrod(u):
    """
    """
    
    B = u.shape[0]
    # torch.norm( )
    theta = torch.linalg.norm(u, dim=1).reshape(-1,1,1).float()
    unorm = F.normalize(u)
    unorm_skew = torch.zeros(B,3,3)
    unorm_skew[:,0,1] = -unorm[:,2] 
    unorm_skew[:,0,2] = unorm[:,1] 
    unorm_skew[:,1,2] = -unorm[:,0] 
    unorm_skew[:,1,0] = unorm[:,2] 
    unorm_skew[:,2,0] = -unorm[:,1] 
    unorm_skew[:,2,1] = unorm[:,0] 
    
    I = torch.zeros(B,3,3)
    I[:,0,0] = 1.0
    I[:,1,1] = 1.0
    I[:,2,2] = 1.0
    
    torch.sin(theta) * unorm_skew
    R = I + torch.sin(theta) * unorm_skew + (1 - torch.cos(theta)) * unorm_skew @ unorm_skew
    
    return R.permute(0, 2, 1)


u = np.random.rand(6)
u = u.reshape(2,3)



print(utils.get_rot_matrix(u[0,:])[0].T)
print(utils.get_rot_matrix(u[1,:])[0].T)
print(compute_rotation_matrix_from_eulerrod(torch.tensor(u)))