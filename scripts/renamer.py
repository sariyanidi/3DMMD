#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:24:07 2023

@author: sariyanide
"""

import os
import sys
from glob import glob

fov = 20
# which_bfm = 'BFMmm-19830'
which_bfm = 'BFMmm-23660'

# sdir = f'/online_data/face/yt_faces2/3DI/OD0.6_OE1.7_OB0.7_RS78_IN0_SF0_CF0.42NF7_NTC7_UL0_CB1GLOBAL79_IS0K7440.75P1{which_bfm}/{fov}' #sys.argv[1]
sdir = f'/online_data/face/yt_faces2/3DI/OD0.6_OE1.7_OB2.7_RS78_IN0_SF0_CF0.42NF7_NTC7_UL0_CB1GLOBAL79_IS0K7440.75P1{which_bfm}/{fov}'
print(sdir)
ddir = '/online_data/face/combined_celeb_ytfaces'
#%%

files = glob(f'{sdir}/*vars')

for f in files:
    bn = os.path.basename(f)
    bn = bn[:9]
    
    print(bn)
    tf = f'{ddir}/{bn}.label{fov}{which_bfm}'
    cmd = f'cp {f} {tf}'
    print(cmd)
    os.system(cmd)




#%%
sdir = f'/online_data/face/CelebAMask-HQ/3DI/OD0.6_OE1.7_OB0.7_RS78_IN0_SF0_CF0.42NF1_NTC7_UL0_CB1GLOBAL79_IS0K7440.75P1{which_bfm}/{fov}'
files = glob(f'{sdir}/*vars')



for f in files:
    bn = os.path.basename(f).split('.')[0].split('_')[0]
    # bn = bn[:9]
    
    print(bn)
    tf = f'{ddir}/{bn}.label{fov}{which_bfm}'
    cmd = f'cp {f} {tf}'
    print(cmd)
    os.system(cmd)





#%%
sdir = f'/online_data/face/img_align_celeba/3DI/OD0.6_OE1.7_OB0.7_RS78_IN0_SF0_CF0.42NF1_NTC7_UL0_CB1GLOBAL79_IS0K7440.75P1{which_bfm}/{fov}'
files = glob(f'{sdir}/*vars')

ddir = '/online_data/face/combined_celeb_ytfaces'


for f in files:
    bn = os.path.basename(f).split('.')[0].split('_')[0]
    # bn = bn[:9]
    
    tf = f'{ddir}/{bn}.label{fov}{which_bfm}'
    
    if os.path.exists(tf):
        continue
    print(bn)
    
    cmd = f'cp {f} {tf}'
    # print(cmd)
    os.system(cmd)



