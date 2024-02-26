#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:43:23 2024

@author: sariyanide
"""
import os
import numpy as np
from glob import glob
sdir = '/offline_data/face/CAR/SocialCoordination/3DID_output/TrueTrue'
ddir = '/offline_data/face/CAR/SocialCoordination/3DID_output/txts'

if not os.path.exists(ddir):
    os.mkdir(ddir)

files = glob(f'{sdir}/*npy')

for f in files:
    bn = os.path.basename(f)
    newf = os.path.join(ddir, bn.replace('_3DID.npy', 'pe3DID'))
    if os.path.exists(newf):
        continue
    try:
        obj = np.load(f, allow_pickle=True).item()
        etl = obj['exps']
        ptl = obj['angles']
        e = np.array([et.numpy() for et in etl]).squeeze()
        p = np.array([pt.numpy() for pt in ptl]).squeeze()
        x = np.concatenate((p, e), axis=1)
        
        np.savetxt(newf, x)
    except:
        print('Skipping: ', newf)


