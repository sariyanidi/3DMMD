#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:57:11 2023

@author: v
"""

import torch
import numpy as np

class Camera():
    
    def __init__(self, fov_x, fov_y, cx, cy):
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.cx = cx
        self.cy = cy
        
        self.f_x = self.cx/(np.tan(np.deg2rad(self.fov_x)/2.0))
        self.f_y = self.cy/(np.tan(np.deg2rad(self.fov_y)/2.0))
        
    def get_matrix(self):
        return np.array([self.f_x, 0, self.cx, 0, self.f_y, self.cy, 0, 0, 1]).reshape(3,3)
    
    def map_to_2d(self, mesh):
        x = self.f_x*mesh[:,:,0:1]/mesh[:,:,2:]+self.cx
        y = self.f_y*mesh[:,:,1:2]/mesh[:,:,2:]+self.cy
        p = torch.cat((x,y), dim=2)
        
        return p
        