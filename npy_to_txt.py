#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:43:23 2024

@author: sariyanide
"""
import os
import sys
import numpy as np
from glob import glob
sdir = '/offline_data/face/CAR/SocialCoordination/3DID_output/TrueTrue'
ddir = '/offline_data/face/CAR/SocialCoordination/3DID_output/txts'

if not os.path.exists(ddir):
    os.mkdir(ddir)

files = glob(f'{sdir}/*npy')

for f in files:
    bn = os.path.basename(f)
    newf = os.path.join(ddir, bn.replace('_undistorted', '').replace('_3DID.npy', '.pe3DID'))
    
    if os.path.exists(newf):
        continue
    try:
        obj = np.load(f, allow_pickle=True).item()
        etl = obj['exps']
        ptl = obj['angles']
        
        erows = []
        for erow in etl:
            if erow is not None:
                erows.append(erow.cpu().numpy())
            else:
                erows.append(np.nan*np.ones((1, 79)))
                
        prows = []
        for prow in ptl:
            if prow is not None:
                prows.append(prow.cpu().numpy())
            else:
                prows.append(np.nan*np.ones((1, 3)))
                
        e = np.array(erows).squeeze()
        p = np.array(prows).squeeze()
        # e = np.array([et.numpy() for et in etl]).squeeze()
        # p = np.array([pt.numpy() for pt in ptl]).squeeze()
        x = np.concatenate((p, e), axis=1)
        
        # break
        np.savetxt(newf, x)
    except Exception as e:
        print('Skipping: ', newf)
        print(e)

#%%
sys.exit(0)

sdir = '/offline_data/face/CAR/SocialCoordination/BFMmm-19830.cfg1.global4/'


files = glob(f'{sdir}/*pe3DID')

for f in files:
    speech_file = f.replace('pe3DID', 'speech_labels')
    if not os.path.exists(speech_file):
        continue
    x = np.loadtxt(f)
    s = np.loadtxt(speech_file).reshape(-1,1)
    T = min(x.shape[0], s.shape[0])
    sx = np.concatenate((s[:T,:], x[:T,:]), axis=1)
    
    spe_file = f.replace('pe3DID', 'spe3DID')
    np.savetxt(spe_file, sx)
