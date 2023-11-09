#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:51:13 2023

@author: v
"""
import os
import sys
import numpy as np
from glob import glob
sys.path.append('../')

import matplotlib.pyplot as plt
from models import morphable_model, orthograhic_fitter



from torch.utils.data import Dataset
from torchvision.io import read_image

class SimpleDataset(Dataset):
    
    def __init__(self, rootdir='/media/v/SSD1TB/dataset/for_3DID/cropped/', is_train=True):
        self.rootdir = rootdir
        all_label_paths = glob(f'{self.rootdir}/*txt')
        all_label_paths.sort()
        
        Ntot = len(all_label_paths)
        Ntra = int(0.75*Ntot)
        if is_train:
            self.label_paths = all_label_paths[:Ntra]
        else:
            self.label_paths = all_label_paths[Ntra:]
        

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        label_fpath = self.label_paths[idx]
        img_fpath = label_fpath.replace('.txt', '.jpg')
        image = read_image(img_fpath)
        label = np.loadtxt(label_fpath)
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label
    
    def create_labels(self):
        
        from scipy.spatial.transform import Rotation
        from utils import utils
        from glob import glob

        model = morphable_model.MorphableModel()
        fitter = orthograhic_fitter.OrthographicFitter(model)
        
        files = glob(f'{self.rootdir}/*txt')
        files.sort()
        
        Y = []
        
        for f in files:
            print(f)
            label_path = f.replace('txt', 'label0')
            
            if os.path.exists(label_path):
                continue
            lmks = np.loadtxt(f)[17:,:]
            fit_params, _ = fitter.fit_orthographic_GN(lmks)
            R, _ = utils.get_rot_matrix(fit_params['u'])
            
            r = Rotation.from_matrix(R)
            angles = r.as_euler('zxy')
            vec = angles.tolist()+[fit_params['taux'], fit_params['tauy'], fit_params['sigmax'], fit_params['sigmay']]
            vec = np.array(vec)
            
            np.savetxt(label_path, vec)

dataset = SimpleDataset()


