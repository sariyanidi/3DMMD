#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:57:11 2023

@author: v
"""

import numpy as np

class Camera():
    
    def __init__(self, fov_x=20.0, fov_y=20.0, cx=112.0, cy=112.0):
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.cx = cx
        self.cy = cy
        
        self.f_x = self.cx/(np.tan(np.deg2rad(self.fov_x)/2.0))
        self.f_y = self.cy/(np.tan(np.deg2rad(self.fov_y)/2.0))
        
    def get_matrix(self):
        return np.array([self.f_x, 0, self.cx, 0, self.f_y, self.cy, 0, 0, 1]).reshape(3,3)
        
        