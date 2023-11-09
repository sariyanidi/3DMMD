#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:27:23 2023

@author: v
"""

import os
import torch
import torch.nn as nn

# from resnets import resnet18, conv1x1

from models import resnets



def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict



class SimpleModel(nn.Module):
    
    resnet18_last_dim = 512
    
    def __init__(self, use_last_fc=False, init_path='./models/checkpoints/resnet18-f37072fd.pth'):
        
        super().__init__()
        self.use_last_fc = use_last_fc

        state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
        self.backbone = resnets.resnet18()
        self.backbone.load_state_dict(state_dict)
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                # conv1x1(self.resnet18_last_dim, 80, bias=True), # id layer
                # conv1x1(self.resnet18_last_dim, 64, bias=True), # exp layer
                # conv1x1(self.resnet18_last_dim, 80, bias=True), # tex layer
                resnets.conv1x1(self.resnet18_last_dim, 3, bias=True),  # angle layer
                # conv1x1(self.resnet18_last_dim, 27, bias=True), # gamma layer
                resnets.conv1x1(self.resnet18_last_dim, 2, bias=True),  # tx, ty
                resnets.conv1x1(self.resnet18_last_dim, 1, bias=True),   # sigmax
                resnets.conv1x1(self.resnet18_last_dim, 1, bias=True)   # sigmay
            ])
            # for m in self.final_layers:
            #     nn.init.constant_(m.weight, 0.)
            #     nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        return x
    
    
    
    
    
    
    
    
    
    
    