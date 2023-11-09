#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 08:37:41 2023

@author: v
"""

from time import time
import copy
# from mesh_renderer import MeshRenderer
import torch.nn.functional as F

import os
import torch
import numpy as np


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


class MorphableModel():
    
    
    def __init__(self, key='BFMmm-19830', data_rootdir='./data', device='cuda',
                 im_w=512.0, im_h=512.0):
        self.data_dir = f'{data_rootdir}/{key}'
                
        self.device = device
        X0 = torch.from_numpy(np.loadtxt(f'{self.data_dir}/X0_mean.dat')).unsqueeze(1).T # [1, N]
        Y0 = -torch.from_numpy(np.loadtxt(f'{self.data_dir}/Y0_mean.dat')).unsqueeze(1).T # [1, N]
        Z0 = torch.from_numpy(np.loadtxt(f'{self.data_dir}/Z0_mean.dat')).unsqueeze(1).T # [1, N]
        self.li = torch.from_numpy(np.loadtxt(f'{self.data_dir}/lmks.dat')).type(torch.int64).to(self.device)

        self.N = X0.shape[1]

        # Texture basis
        self.T = torch.from_numpy(np.loadtxt(f'{self.data_dir}/TEX.dat')).float().to(self.device) # [1, N]

        self.sigma_alphas = torch.from_numpy(np.loadtxt(f'{self.data_dir}/sigma_alphas.dat')).float().to(self.device) # [1, N]
        self.sigma_betas = torch.from_numpy(np.loadtxt(f'{self.data_dir}/sigma_alphas.dat')).float().to(self.device) # [1, N]
        
        self.mean_shape = torch.cat((X0, Y0, Z0), axis=0).T.reshape(-1,1).float().to(self.device)
        self.mean_tex   = torch.from_numpy(np.loadtxt(f'{self.data_dir}/tex_mu.dat')).reshape(-1,1).float().to(self.device)
        
        p0L = copy.deepcopy(self.mean_shape).reshape(-1,3)[self.li,:]
        xmean, ymean, zmean = p0L.mean(axis=0)#reshape(1,3)
        # print(xmean, ymean, zmean)
        self.mean_shape[::3] -= xmean
        self.mean_shape[1::3] -= ymean
        self.mean_shape[2::3] -= zmean

        self.tri = torch.from_numpy(np.loadtxt(f'{self.data_dir}/tl.dat')-1).type(torch.int64).to(self.device) # [M, 3]
        self.point_buf = self.get_point_buf()
        
        IX = torch.from_numpy(np.loadtxt(f'{self.data_dir}/IX.dat')).unsqueeze(2).float().to(self.device) # [N, Kid, 1]
        IY = -torch.from_numpy(np.loadtxt(f'{self.data_dir}/IY.dat')).unsqueeze(2).float().to(self.device) # [N, Kid, 1]
        IZ = torch.from_numpy(np.loadtxt(f'{self.data_dir}/IZ.dat')).unsqueeze(2).float().to(self.device) # [N, Kid, 1]
        
        EX = torch.from_numpy(np.loadtxt(f'{self.data_dir}/E/EX_79.dat')).unsqueeze(2).float().to(self.device) # [N, Kexp, 1]
        EY = -torch.from_numpy(np.loadtxt(f'{self.data_dir}/E/EY_79.dat')).unsqueeze(2).float().to(self.device) # [N, Kexp, 1]
        EZ = torch.from_numpy(np.loadtxt(f'{self.data_dir}/E/EZ_79.dat')).unsqueeze(2).float().to(self.device) # [N, Kexp, 1]
        
        self.Kid = IX.shape[1]
        self.Kexp = EX.shape[1]
        self.Ktex = self.T.shape[1]
        
        # Identity shape basis
        self.I = torch.cat((IX, IY, IZ), dim=2).permute(0,2,1).reshape(-1, self.Kid)
        
        # Expression shape basis
        self.E = torch.cat((EX, EY, EZ), dim=2).permute(0,2,1).reshape(-1, self.Kexp)        
        
        self.SH = SH()
        init_lit = np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0])
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)

        
    
    def compute_face_shape(self, alpha, eps=None):
        """
        Compute 3D mesh from 3DMM shape and (optionally) expression coefficients
        
        @param alpha    Identity shape coefficients; a matrix of shape [B, Kid]
        @param eps      Expresion shape coefficients; a matrix of shape [B, Kexp]
        
        @return 3D mesh; a matrix of shape [B, 3N]
        """
        B = alpha.shape[0]
        dface_shape = (alpha @ self.I.T)
       
        if eps is not None:
            dface_shape += (eps @ self.E.T)
        
        face = dface_shape + self.mean_shape.reshape(1, -1)
        return face.reshape(B, -1, 3) 
    
    
    def compute_rotation_matrix(self, angles):
        """
        Compute rotation matrix from Euler angles
        
        @param angles   Tensor of shape [B, 3], which contains euler angles in radians
        
        @return R       Tensor of size [B, 3, 3], which contains rotation matrices corresponding
                        to each angle. Output is transposed (see comment at return) 
        """
        
        B = angles.shape[0]
        x = angles[:,0].reshape(B,1,1)
        y = angles[:,1].reshape(B,1,1)
        z = angles[:,2].reshape(B,1,1)
        
        ones = torch.ones(B,1,1).to(self.device)
        zeros = torch.zeros(B,1,1).to(self.device)
        
        cosx = torch.cos(x)
        sinx = torch.sin(x)

        cosy = torch.cos(y)
        siny = torch.sin(y)

        cosz = torch.cos(z)
        sinz = torch.sin(z)
        
        Rx_0 = torch.cat((ones, zeros, zeros), axis=2)
        Rx_1 = torch.cat((zeros, cosx, -sinx), axis=2)
        Rx_2 = torch.cat((zeros, sinx, cosx), axis=2)
        
        Ry_0 = torch.cat((cosy, zeros, siny), axis=2)
        Ry_1 = torch.cat((zeros, ones, zeros), axis=2)
        Ry_2 = torch.cat((-siny, zeros, cosy), axis=2)
        
        Rz_0 = torch.cat((cosz, -sinz, zeros), axis=2)
        Rz_1 = torch.cat((sinz, cosz, zeros), axis=2)
        Rz_2 = torch.cat((zeros, zeros, ones), axis=2)
        
        Rx = torch.cat((Rx_0, Rx_1, Rx_2), dim=1)
        Ry = torch.cat((Ry_0, Ry_1, Ry_2), dim=1)
        Rz = torch.cat((Rz_0, Rz_1, Rz_2), dim=1)
        
        R = Rz @ Ry @ Rx
        
        # We return transposed because we'll right-multiply with a mesh of size [N, 3]
        #   (instead of left-multiplying with mesh of [3, N])
        return R.permute(0, 2, 1)
    
    
    def compute_rotation_matrix_from_eulerrod(self, u):
        """
        """
        
        B = u.shape[0]
        # torch.norm( )
        theta = torch.linalg.norm(u, dim=1).reshape(-1,1,1).float()
        unorm = F.normalize(u)
        unorm_skew = torch.zeros(B,3,3).to(self.device)
        unorm_skew[:,0,1] = -unorm[:,2] 
        unorm_skew[:,0,2] = unorm[:,1] 
        unorm_skew[:,1,2] = -unorm[:,0] 
        unorm_skew[:,1,0] = unorm[:,2] 
        unorm_skew[:,2,0] = -unorm[:,1] 
        unorm_skew[:,2,1] = unorm[:,0] 
        
        I = torch.zeros(B,3,3).to(self.device)
        I[:,0,0] = 1.0
        I[:,1,1] = 1.0
        I[:,2,2] = 1.0
        
        torch.sin(theta) * unorm_skew
        R = I + torch.sin(theta) * unorm_skew + (1 - torch.cos(theta)) * unorm_skew @ unorm_skew
        
        return R.permute(0, 2, 1)


        # unorm_skew = skew(unorm)
    
        """
        I = np.eye(3)
        R = I + np.sin(theta) * unorm_skew + (1 - np.cos(theta)) * np.dot(unorm_skew, unorm_skew)

        x = angles[:,0].reshape(B,1,1)
        y = angles[:,1].reshape(B,1,1)
        z = angles[:,2].reshape(B,1,1)
        
        ones = torch.ones(B,1,1).to(self.device)
        zeros = torch.zeros(B,1,1).to(self.device)
        
        cosx = torch.cos(x)
        sinx = torch.sin(x)

        cosy = torch.cos(y)
        siny = torch.sin(y)

        cosz = torch.cos(z)
        sinz = torch.sin(z)
        
        Rx_0 = torch.cat((ones, zeros, zeros), axis=2)
        Rx_1 = torch.cat((zeros, cosx, -sinx), axis=2)
        Rx_2 = torch.cat((zeros, sinx, cosx), axis=2)
        
        Ry_0 = torch.cat((cosy, zeros, siny), axis=2)
        Ry_1 = torch.cat((zeros, ones, zeros), axis=2)
        Ry_2 = torch.cat((-siny, zeros, cosy), axis=2)
        
        Rz_0 = torch.cat((cosz, -sinz, zeros), axis=2)
        Rz_1 = torch.cat((sinz, cosz, zeros), axis=2)
        Rz_2 = torch.cat((zeros, zeros, ones), axis=2)
        
        Rx = torch.cat((Rx_0, Rx_1, Rx_2), dim=1)
        Ry = torch.cat((Ry_0, Ry_1, Ry_2), dim=1)
        Rz = torch.cat((Rz_0, Rz_1, Rz_2), dim=1)
        
        R = Rz @ Ry @ Rx
        
        # We return transposed because we'll right-multiply with a mesh of size [N, 3]
        #   (instead of left-multiplying with mesh of [3, N])
        return R.permute(0, 2, 1)
        """

    
    def view_transform(self, mesh, R, tau):
        """
        View-transform a given mesh. That is, we rotate and translate using R and tau
        
        @param mesh     A tensor of size [B, N, 3], which contains the face meshes
        @param R        A tensor of size [B, 3, 3], which contains rotation matrices
        @param tau      A tensor of size [B, 3], containing translation vectors
        
        @return         View-transformed mesh
        """
        return mesh @ R + tau.unsqueeze(1)
    
    
    def compute_texture(self, beta):
        # B = alpha.shape[0]
        return (beta @ self.T.T).unsqueeze(2) + self.mean_tex.reshape(1, -1, 1)


    def get_point_buf(self):
        """
        Get point buffer. Create if it's not in the right filepath
        
        @return point buffer
        
        """
        
        filepath =  f'{self.data_dir}/point_buf.dat'
        
        if os.path.exists(filepath):
            return torch.from_numpy(np.loadtxt(filepath)).type(torch.int64)
        
        N = self.tri.max()+1
        uixs = []
        lens = []
        for i in range(N):
            ix = (self.tri == i).type(torch.int64) * torch.arange(1, self.tri.shape[0]+1).reshape(-1, 1)
            uix = torch.unique(ix)-1
            uix = uix[torch.where(uix>=0)]
            lens.append(len(uix))
            uixs.append(uix.tolist())
        
        L = max(lens)
        
        for i in range(N):
            if len(uixs[i]) < L:
                for j in range(L-len(uixs[i])):
                    uixs[i].append(uixs[i][-1])
        
        point_buf = torch.tensor(uixs)        
        np.savetxt(filepath, point_buf.cpu().numpy())
        
        return point_buf


    def compute_face_normals(self, mesh):
        """
        Compute face normals for each mesh point
        
        @param mesh     Face mesh, a tensor of size [B, N, 3]
        @return         A tensor of size [B, N, 3] that contains
        """
        
        v1 = mesh[:, self.tri[:, 0]]
        v2 = mesh[:, self.tri[:, 1]]
        v3 = mesh[:, self.tri[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_normals = torch.linalg.cross(e1, e2, dim=-1)
        face_normals = F.normalize(face_normals, dim=-1, p=2)
        face_normals = torch.cat([face_normals, torch.zeros(face_normals.shape[0], 1, 3).to(self.device)], dim=1)
        vertex_norm = torch.sum(face_normals[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm
    
    
    def compute_pixel_values(self, face_texture, face_normals, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_normals[..., :1]).to(self.device),
            -a[1] * c[1] * face_normals[..., 1:2],
             a[1] * c[1] * face_normals[..., 2:],
            -a[1] * c[1] * face_normals[..., :1],
             a[2] * c[2] * face_normals[..., :1] * face_normals[..., 1:2],
            -a[2] * c[2] * face_normals[..., 1:2] * face_normals[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_normals[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_normals[..., :1] * face_normals[..., 2:],
            0.5 * a[2] * c[2] * (face_normals[..., :1] ** 2  - face_normals[..., 1:2] ** 2)
        ], dim=-1)
        
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color




#%%




"""


model = MorphableModel(device='cuda')







# def get_perspective_projection(fov_x, fov_y, c_x, c_y):
#     f_x = c_x/(torch.tan(fov_x/2.0))
#     f_y = c_y/(torch.tan(fov_y/2.0))
    
#     return np.array([f_x, 0, c_x, 
#                      0, f_y, c_y,
#                      0, 0, 1.0]).reshape(3, 3)



# if __name__ == '__main__':


import matplotlib.pyplot as plt


sdir = '/media/v/SSD1TB/dataset/for_3DID/cropped'

lmks = np.loadtxt(f'{sdir}/id003_003.txt')[17:,:]

sdir = '/media/v/SSD1TB/dataset/for_3DID/cropped'
fitter = OrthographicFitter(model)
fit_params, _ = fitter.fit_orthographic_GN(lmks)

R, _ = get_rot_matrix(fit_params['u'])



fov_x = 50
fov_y = 50


alpha0 = torch.zeros(1, model.Kid)
alpha1 = torch.zeros(1, model.Kid)
alpha1[0,0] = 3*model.sigma_alphas[0]
alpha = 0*torch.cat((alpha0, alpha1), dim=0).float().to(model.device)


beta0 = torch.zeros(1, model.Ktex)
beta1 = torch.zeros(1, model.Ktex)
beta1[0,0] = 3*model.sigma_betas[0]/20.
beta = torch.cat((beta0, beta1), dim=0).float().to(model.device)

B = alpha.shape[0]

ps = model.compute_face_shape(alpha)



# plt.figure(figsize=(13,13))
# plt.plot(ps[0,:,0], -ps[0,:,1], '.')
# plt.plot(ps[1,:,0], -ps[1,:,1], '.')

# angles = torch.randn((B, 3)).float()
angles = 0*torch.tensor([0, -np.pi/4, 0, 0, np.pi/4, 0]).reshape(B, 3).float().to(model.device)
R = model.compute_rotation_matrix(angles)
# tau = torch.zeros((B, 3)).float()

tau = torch.tensor([0, 0, 300, 0, 0, 300]).reshape(B, 3).float().to(model.device)

v = model.view_transform(ps, R, tau)
q = model.compute_face_normals(v)


tex = model.compute_tex(beta/3.)

mr = MeshRenderer(rasterize_fov=fov_x, znear=0.001, zfar=100000)

# v[:,:,1] *= -1
# v[:,:,2] *= -1


# x = v[0,model.li, 1]
# y = v[0,model.li, 2]
# ux = -15*q[0,model.li, 1]
# vy = -15*q[0,model.li, 2]

vc = v.cpu()[0,::5,:]

mask, depth, image = mr.forward(v, model.tri, tex)

d0 = depth[0,0,:,:].cpu()
d1 = depth[0,0,:,:].cpu()
plt.imshow(d0)
plt.imshow(d1)

# image[mask == 0] = 0
i0 = image[0,0,:,:].cpu()
i1 = image[1,0,:,:].cpu()

i0 = (i0-i0.min())/(i0.max()-i0.min())
i1 = (i1-i1.min())/(i1.max()-i1.min())

# i0 /= i0.max()
# i1 /= i1.max()

i1[(mask[1]==0)[0,:,:]] = 0

# plt.imshow(i0, cmap='gray')
plt.imshow(i1, cmap='gray')

print(d0.max())
print(d1.max())

#%%

        # vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        # vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        # return vertex_norm


#%%
plt.figure(figsize=(13,13))
plt.plot(v[0,:,0], v[0,:,1], '.')
plt.plot(v[1,:,0], v[1,:,1], '.')
# plt.plot(p[::3,0], -p[1::3,0], '.')

#%%
"""