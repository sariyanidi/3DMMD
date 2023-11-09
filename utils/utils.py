#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 07:26:18 2023

@author: v
"""

import numpy as np


def get_rot_matrix(u):
    # if u.shape[0] == 1:
        # u = u.T
    u = u.reshape(-1,1)
    # print(u.shape)

    theta = np.linalg.norm(u)
    unorm = u / theta
    unorm_skew = skew(unorm)

    I = np.eye(3)
    R = I + np.sin(theta) * unorm_skew + (1 - np.cos(theta)) * np.dot(unorm_skew, unorm_skew)

    IR = I - R

    R1 = np.zeros((3, 3, 3))
    R1[:, :, 0] = u[0] * skew(u)
    R1[:, :, 1] = u[1] * skew(u)
    R1[:, :, 2] = u[2] * skew(u)

    R2 = np.zeros((3, 3, 3))
    R2[:, :, 0] = (IR @ I[:, 0:1]) @ u.T
    R2[:, :, 1] = (IR @ I[:, 1:2]) @ u.T
    R2[:, :, 2] = (IR @ I[:, 2:3]) @ u.T
    
    
    R3 = np.zeros((3,3,3));
    R3[:,:,0] = R1[:,:,0]+R2[:,:,0]-(R2[:,:,0]).T
    R3[:,:,1] = R1[:,:,1]+R2[:,:,1]-(R2[:,:,1]).T
    R3[:,:,2] = R1[:,:,2]+R2[:,:,2]-(R2[:,:,2]).T
    
    
    dR_du1 = (theta ** -2) * (R3[:, :, 0] @ R).reshape(3,3,1)
    dR_du2 = (theta ** -2) * (R3[:, :, 1] @ R).reshape(3,3,1)
    dR_du3 = (theta ** -2) * (R3[:, :, 2] @ R).reshape(3,3,1)

    u1, u2, u3 = u

    dR_du = np.concatenate((dR_du1, dR_du2, dR_du3), axis=2)

    return R, dR_du #, d2R_d11, d2R_d12, d2R_d13, d2R_d22, d2R_d23, d2R_d33



def Uij(i, j):
    U = np.zeros((3, 3))
    U[i, j] = 1
    return U


def skew(u):
    Ux = np.array([[0, -u[2,0], u[1,0]], [u[2,0], 0, -u[0,0]], [-u[1,0], u[0,0], 0]])
    return Ux
