#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:50:21 2023

@author: sariyanide
"""

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
import os
from math import floor, ceil

import random
import sys
# sys.path.append('../')

from utils import utils

# sdir = '/offline_data/face/yt_faces2/images/'#sys.argv[1]
# ddir = '/online_data/face/yt_faces2/yt_cropped2/'#sys.argv[2]

# ddir = '/online_data/face/img_align_celeba/cropped_celeba'


sdir = '/online_data/face/img_align_celeba/images'
ddir = '/online_data/face/combined_celeb_ytfaces'

os.makedirs(ddir, exist_ok=True)


imfs = glob(f'{sdir}/*jpg')
imfs.sort()
# random.shuffle(imfs)
for imf in imfs:

    try:

        bimf = os.path.basename(imf)
        dstim_path = f'{ddir}/{bimf}'

        if os.path.exists(dstim_path):
            continue

        im = cv2.imread(imf)
        im = cv2.imread(imf)
        lf = imf.replace('.jpg', '.txt')
        
        if not os.path.exists(lf):
            continue
        
        lmks = np.loadtxt(lf)
        xs = lmks[::2]
        ys = im.shape[0]-lmks[1::2]
        lmks68 = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)
        M = utils.estimate_norm(lmks68[17:,:], im.shape[0], 1.5, [25,25])
        cim = utils.resize_n_crop_cv(im, M, 224)
        
        cv2.imwrite(dstim_path, cim)
        
        np.savetxt(f'{ddir}/{bimf.replace(".jpg", ".txt")}', lmks)
    except:
        print(f'skipping {bimf} ')

    # break

