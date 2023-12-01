#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:21:00 2023

@author: sariyanide
"""
import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class DirectDataset(Dataset):
    
    def __init__(self,  fov, rasterize_size,
                 rootdir='./cropped_dataset/', 
                 is_train=True,
                 transform=None, normalize_labels=False,
                 which_bfm='BFMmm-19830', 
                 cfgid=2,
                 do_tform=False):
        self.rootdir = rootdir
        # self.extn = 'label0000'
        self.fov = fov
        self.rasterize_size = rasterize_size
        self.extn = f'label_post{self.fov}-{cfgid}-{which_bfm}'
        self.extn_pre = f'label{self.fov}-{cfgid}-{which_bfm}'
        self.extn_euler = f'label_eulerrod2{self.fov:.2f}'
        self.normalize_labels = normalize_labels
        self.do_tform = do_tform
        all_label_paths = glob(f'{self.rootdir}/*{self.extn}')
        all_prelabel_paths = glob(f'{self.rootdir}/*{self.extn_pre}')
        print(self.extn)
        print(len(all_label_paths))
        print(len(all_prelabel_paths))
        
        if len(all_label_paths) < len(all_prelabel_paths):
            all_prelabel_paths = glob(f'{self.rootdir}/*{self.extn_pre}')
            
            for f in all_prelabel_paths:
                all_label_paths.append(self.process_and_save_label(f))

        all_label_paths.sort()
        
        self.transform = transform
        
        Ntot = len(all_label_paths)
        Ntra = int(0.96*Ntot)
        
        self.stds = None
        self.means = None
        
        self.A = None
        if self.normalize_labels:
            all_labels = []
            for f in all_label_paths[:Ntra]:
                all_labels.append(np.loadtxt(f))
            self.A = np.array(all_labels)
            self.stds = np.std(self.A, axis=0).astype(np.float32)
            self.means = np.mean(self.A, axis=0).astype(np.float32)
            # print(A)
        if is_train:
            self.label_paths = all_label_paths[:Ntra]
        else:
            self.label_paths = all_label_paths[Ntra:]
        
                
        self.tforms = v2.Compose([v2.RandomAffine(1.4, (0,0.01), (0.98, 1.02)),
                                  v2.RandomPosterize(4, 0.3),
                                  v2.RandomAdjustSharpness(2),
                                  v2.RandomAutocontrast(0.2),
                                  v2.RandomEqualize(0.2),
                                  v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.2)),
                                  #v2.ColorJitter(brightness=[0.1, 0.5], hue=[0.1, 0.4], 
                                  #               saturation=[0.1, 0.5])
                                  ])
                
        # posterizer = v2.RandomPosterize(bits=2)
        # solarizer = v2.RandomSolarize(threshold=192.0)
        # sharpness_adjuster = v2.RandomAdjustSharpness(sharpness_factor=2)
        # sharpness_adjuster = v2.RandomAdjustSharpness(sharpness_factor=2)
        # autocontraster = v2.RandomAutocontrast()
        # equalizer = v2.RandomEqualize()

        # translate: Optional[Sequence[float]] = None,
        # scale: Optional[Sequence[float]] = None,
        # shear: Optional[Union[int, float, Sequence[float]]] = None,
        # interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        # fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = 0,
        # center: Optional[List[float]] = None,
        
        # RandomAutocontrast([p])
        # RandomAffine()
        # RandomAutocontrast([p])
        
        # v2.GaussianBlur(kernel_size[, sigma])
        
        # v2.RandomPhotometricDistort([brightness, ...])
        
        # v2.RandomSolarize(threshold[, p])
        
        # v2.RandomEqualize([p])
        
        
        
    def ntot_samples(self):
        return len(self.label_paths)


    def __len__(self):
        return len(self.label_paths)
    
    
    
    def check_one_by_one(self):
        for n in range(len(self.label_paths)):
            print(os.path.basename(self.label_paths[n]))
            self.__getitem__(n)
    
    

    def process_and_save_label(self, prelabel_fpath):
        label = np.loadtxt(prelabel_fpath)
        # rigid_label = np.loadtxt(rigid_label_fpath)
        
        alpha = label[:199]
        beta = label[199:2*199]
        exp = label[(2*199+6):(2*199+6+79)]
    
        taus = label[2*199:(2*199+3)]
        us = label[(2*199+3):(2*199+6)]
        
        taus[1] *= -1
        us[0] *= -1
        us[-1] *= -1

        y = np.concatenate((alpha, exp, beta, np.zeros(27), us, taus), axis=0)
        
        label_fpath = prelabel_fpath.replace(self.extn_pre, self.extn)
        np.savetxt(label_fpath, y)
        
        return label_fpath
        

    def __getitem__(self, idx):
        label_fpath = self.label_paths[idx]
        img_fpath = label_fpath.replace(f'.{self.extn}', '.jpg')

        image = read_image(img_fpath).float()/255.0
        label = np.loadtxt(label_fpath)
        
        
        # print(label)
        # print(image.shape)
        
        # rigid_label = np.loadtxt(rigid_label_fpath)
        
        
        # print(f'{us} vs {us0}')
        
        # # label[-1] -= 1000
        # lmks = np.loadtxt(lmks_fpath)
        # tlmks = copy.deepcopy(lmks)
        # tlmks[:,1] = self.rasterize_size-tlmks[:,1]
        # rigid_tform = utils.estimate_norm(tlmks[17:,:], self.rasterize_size)
        
        if self.transform:
            image = self.transform(image)
        if self.normalize_labels:
            # Knonrigid  = 199*2+79
            label -= self.means
            label /= (self.stds+1e-10)
        
        if self.do_tform:
            image = self.tforms(image)
            
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, torch.from_numpy(label).float()
    