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
    
    def __init__(self, fovx, fovy, cx, cy, us, taus, alpha, Nframes=7, 
                 which_bfm='BFMmm-23660'):
         super().__init__()
         self.Nframes = Nframes
         self.mm = morphable_model.MorphableModel(key=which_bfm)
         self.camera = camera.Camera(fovx, fovy, cx, cy)
         
         self.alpha = nn.Parameter(alpha)
         self.us = nn.Parameter(us)
         self.taus = nn.Parameter(taus)
        
        
    def init_frame(self, i, u, tau):
        self.us[:,3*i:3*(i+1)] = u
        self.taus[:,3*i:3*(i+1)] = tau
    
        
    def parse_frame(self, i):
        return (self.us[:,3*i:3*(i+1)], self.taus[:,3*i:3*(i+1)])
    
    def compute_face_shape(self):
        return self.mm.compute_face_shape(self.alpha)
    
        
    def gather_targets(self, raw_ps):
        # u = x[:,:3]
        # tau = x[:, 3:6]
        ps = []
        for p in raw_ps:
            p = p.flatten(start_dim=1).unsqueeze(-1)
            ps.append(p)
        
        ps = torch.cat(ps, dim=2)
        ps = torch.flatten(ps, start_dim=1)
        
        return ps
        
    
    def forward(self):
        # u = x[:,:3]
        # tau = x[:, 3:6]
        ps = []
        for n in range(self.Nframes):
            (u, tau) = self.parse_frame(n)
            p = self.mm.project_to_2d(self.camera,  u, tau, self.alpha)
            p = p.flatten(start_dim=1).unsqueeze(-1)
            ps.append(p)
        
        ps = torch.cat(ps, dim=2)
        ps = torch.flatten(ps, start_dim=1)
        
        return ps



if __name__ == '__main__':
    print(123)
    r = MultiframeReconstructor(20, 20, 112, 112, torch.ones([1, 7*3]), torch.ones([1, 7*3]))
    r.to('cuda')
    r()
