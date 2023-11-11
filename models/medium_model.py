#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:27:23 2023

@author: v
"""

import os
import torch
import torch.nn as nn
from models import camera 
from utils import utils
from torch.nn import functional as F

import sys
sys.path.append('../insightface/recognition/')
from arcface_torch.backbones import get_model

# from resnets import resnet18, conv1x1

from models import resnets, morphable_model, mesh_renderer



def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict





class RecogNetWrapper(nn.Module):
    def __init__(self, net_recog, pretrained_path=None, input_size=112):
        super(RecogNetWrapper, self).__init__()
        net = get_model(name=net_recog, fp16=True)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            net.load_state_dict(state_dict)
            print("loading pretrained net_recog %s from %s" %(net_recog, pretrained_path))
        for param in net.parameters():
            param.requires_grad = False
        self.net = net
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
        
    def forward(self, image, M):
        image = self.preprocess(utils.resize_n_crop(image, M, self.input_size))
        id_feature = F.normalize(self.net(image), dim=-1, p=2)
        return id_feature




class MediumModel(nn.Module):
    
    resnet18_last_dim = 512
    resnet50_last_dim = 2048
    
    def __init__(self, 
                 rasterize_fov,
                 rasterize_size,
                 device='cuda',
                 # init_path='./models/checkpoints/resnet18-f37072fd.pth'
                 init_path='./models/checkpoints/resnet50-0676ba61.pth'
                 ):
        
        super().__init__()
        self.mm = morphable_model.MorphableModel()
        self.device = device
        self.use_last_fc = False
        self.rasterize_fov = rasterize_fov
        
        self.cam = camera.Camera(fov_x=rasterize_fov, fov_y=rasterize_fov, 
                                 cx=rasterize_size/2.0, cy=rasterize_size/2.0)
        
        # Renderer requires GPU
        self.renderer = mesh_renderer.MeshRenderer(rasterize_fov, rasterize_size=rasterize_size, use_opengl=False).to(self.device)

        state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
        # self.backbone = resnets.resnet18()
        self.backbone = resnets.resnet50()
        self.backbone.load_state_dict(state_dict)
        
        self.net_recog = RecogNetWrapper(net_recog='r50', 
                                         pretrained_path='./models/checkpoints/recog_model/backbone.pth')
        self.net_recog.to(self.device)
        self.net_recog.net.eval()
        
        if not self.use_last_fc:
            """
            """
            self.MM_layers = nn.ModuleList([
                resnets.conv1x1(self.resnet50_last_dim, 199, bias=True), # id layer
                resnets.conv1x1(self.resnet50_last_dim, 79, bias=True), # exp layer
                resnets.conv1x1(self.resnet50_last_dim, 199, bias=True) # tex layer
            ])
            
            self.illum_layers = nn.ModuleList([
                resnets.conv1x1(self.resnet50_last_dim, 27, bias=True) # gamma layer
            ])
            
            self.rigid_layers = nn.ModuleList([
                resnets.conv1x1(self.resnet50_last_dim, 3, bias=True),  # angle layers
                resnets.conv1x1(self.resnet50_last_dim, 1, bias=True),  # tx, ty, tz
                resnets.conv1x1(self.resnet50_last_dim, 1, bias=True),  # tx, ty, tz
                resnets.conv1x1(self.resnet50_last_dim, 1, bias=True),  # tx, ty, tz
            ])
            
            
            if self.rasterize_fov < 13:
                offset = 1000.0
            elif self.rasterize_fov == 30:
                offset = 510
            elif self.rasterize_fov == 60:
                offset = 230
                
            # nn.init.constant_(self.rigid_layers[-1].bias, 1000.0)
            nn.init.constant_(self.rigid_layers[-1].bias, offset)
            
            
            # for m in self.final_layers:
            #     nn.init.constant_(m.weight, 0.)
            #     nn.init.constant_(m.bias, 0.)

    def forward(self, input_im, tforms, render=False):
        x = self.backbone(input_im)
        if not self.use_last_fc:
            output = []
            for layer in self.MM_layers+self.illum_layers+self.rigid_layers:
            # for layer in self.rigid_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        
        mask = None
        masked_input = None
        rendered_output = None
        lmks = None
        gt_feat = None
        pred_feat = None
        
        if render:
            params = self.parse_params(x)
            # params['tau'][:,-1] = params['tau'][:,-1] + 1000
            (mask, _, rendered_output), p = self.render_image(params)
            masked_input = input_im[:,0:1,:,:]*mask#.repeat([1,3,1,1])
            lmks = p[:,self.mm.li,:]
            # params['tau'][:,-1] = params['tau'][:,-1] - 1000
            
            rendered_output_3ch = rendered_output.repeat(1,3,1,1)
            # print(input_im.shape)
            # print(rendered_output_3ch.shape)
            # print('------------------------------')
            self.net_recog.eval()
            self.net_recog.net.eval()
            assert self.net_recog.training == False
            masked_input = masked_input.repeat(1,3,1,1)
            gt_feat = self.net_recog(masked_input, tforms)
            # pred_feat = self.net_recog(input_im, tforms)
            # tforms = tforms.requires_grad(True)
            # tforms.requires_grad_(True)
            pred_feat = self.net_recog(rendered_output_3ch, tforms)
            
        
        return x, masked_input, rendered_output, mask, lmks, gt_feat, pred_feat
    
    
    def freeze_all_but_rigid_layers(self):

        for param in self.MM_layers.parameters():
            param.requires_grad = False

        for param in self.illum_layers.parameters():
            param.requires_grad = False
    
    
    def freeze_rigid_layers(self):

        for param in self.rigid_layers.parameters():
            param.requires_grad = False    
            
        for param in self.backbone.parameters():
            param.requires_grad = False    
            
    def unfreeze_rigid_layers(self):

        for param in self.rigid_layers.parameters():
            param.requires_grad = True
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    
    def parse_params(self, y):
        return {'alpha': y[:,:199],
                  'exp': y[:, 199:(199+79)],
                  'beta': y[:,(199+79):(199+79+199)],
                  'gamma': y[:,(199*2+79):(199*2+79+27)],
                  'angles': y[:,(199*2+79+27):(199*2+79+27+3)],
                  'tau': y[:,(199*2+79+27+3):]}
        
        # B = y.shape[0]
        # return {'alpha': torch.zeros(B, 199).to(self.device),
        #           'exp': torch.zeros(B, 79).to(self.device),
        #           'beta': torch.zeros(B, 199).to(self.device),
        #           'gamma': torch.zeros(B, 27).to(self.device),
        #           'angles': y, # y[:,(199*2+79+27):(199*2+79+27+3)],
        #           'tau': torch.zeros(B, 3).to(self.device)}
        
        
    
    def render_image(self, params):
        canonical_mesh = self.mm.compute_face_shape(params['alpha'])
        # canonical_mesh[:,:, 1] *= -1
        # R = self.mm.compute_rotation_matrix(params['angles'])
        R = self.mm.compute_rotation_matrix_from_eulerrod(params['angles'])
        
        # view-transformed mesh
        mesh = self.mm.view_transform(canonical_mesh, R, params['tau'])
        texture = self.mm.compute_texture(params['beta'])
        # mesh[:,:, 1] *= -1
        
        # print(mesh.shape)
        p = self.cam.map_to_2d(mesh)
        # print(p.shape)
        # print(mesh.shape)

        return self.renderer(mesh, self.mm.tri, feat=texture), p
        
        # d = y[1].squeeze().detach().cpu().numpy()
        # df = d.flatten()
        # w = np.where(df>0)[0]
        # df = df[w]
        
        # d = (d-df.min())/(df.max()-df.min())
        # d[np.where(d<0)] = 0
        # d[np.where(d>1)] = 1
        
        # im = y[-1].squeeze()[0,:,:].cpu().numpy()
        # plt.imshow(im)
        # plt.plot(lmks[:,0], 224-lmks[:,1], 'x')
        
        
    
    
    def unfreeze_all_but_rigid_layers(self):

        for param in self.MM_layers.parameters():
            param.requires_grad = True

        for param in self.illum_layers.parameters():
            param.requires_grad = True
    
    
    
    
    
    
    