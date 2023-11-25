#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 07:59:26 2023

@author: sariyanide
"""


import os
import torch
import torch.nn as nn
from models import camera 
from utils import utils
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur
from kornia.morphology import erosion


from models import resnets, morphable_model, mesh_renderer, camera


class MultiframeReconstructor(nn.Module):
    def __init__(self, fovx, fovy, cx, cy, which_bfm='BFMmm-23660'):
         super().__init__()
         self.mm = morphable_model.MorphableModel(key=which_bfm)
         self.camera = camera.Camera(fovx, fovy, cx, cy)
         self.alpha = nn.Parameter(self.mm.Kid)
        
    def forward(self, x):
        u = x[:,:3]
        tau = x[:, 3:6]
        
        R = self.mm.compute_rotation_matrix_from_eulerrod(u)
        p = self.camera.map_to_2d()
                
            