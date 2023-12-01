#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:24:07 2023

@author: sariyanide
"""

import os
import sys
from glob import glob

cfgid = 2
fov = 15
which_bfm = 'BFMmm-23660'
sdir = f'/online_data/face/combined_celeb_ytfaces-labels/3DI-{cfgid}-{fov}-{which_bfm}'
dbname = 'celeb' # 'celeb'

if dbname == 'ytfaces':
    files = glob(f'{sdir}/id*vars')
elif dbname == 'celeb':
    files = glob(f'{sdir}/[0-9]*vars')
    

ddir = '/online_data/face/combined_celeb_ytfaces'


for fi, f in enumerate(files):
    if dbname == 'ytfaces':
        bn = os.path.basename(f).split('.')[0].split('_co b')[0]
    elif dbname == 'celeb':
        bn = os.path.basename(f).split('.')[0].split('_')[0]
    
    # bn = bn[:9]
    
    tf = f'{ddir}/{bn}.label{fov}-{cfgid}-{which_bfm}'
    
    if fi % 1000 == 0:
        print(fi)
    
    if os.path.exists(tf):
        continue
    # print(bn)
    
    cmd = f'cp {f} {tf}'
    print(cmd)
    os.system(cmd)
    # break


