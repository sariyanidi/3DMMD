#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 07:55:21 2023

@author: sariyanide
"""
import sys
sys.path.append('./models')

import camera
import video_fitter
import argparse 
import cv2
import os

from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--vid_path', type=str, default='-tc_cond-is_adm-001_n-20200203110206_2_undistorted.mp4')
parser.add_argument('--fov', default=30.0, type=float)
parser.add_argument('--GPUid', default=0, type=int)
parser.add_argument('--first_3DID', default=1, type=int)

sdir = '/offline_data/face/CAR/SocialCoordination/3DI_input/'

args = parser.parse_args()

cap = cv2.VideoCapture(args.vid_path)

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

camera = camera.Camera(args.fov, args.fov, float(width)/2.0, float(height)/2.0)

vf = video_fitter.VideoFitter(camera, outdir_root='/offline_data/face/CAR/SocialCoordination/3DID_output',Tmax=None)

ids = glob(f'{sdir}/*mp4')
ids.sort()
ids = ids[args.GPUid::3]

for id in ids:
    bn = os.path.basename(id).split('.')[0]
    
    vids = glob(f'{sdir}/../preprocessing/{bn}*mp4')
    
    if len(vids) == 0:
        vid_path = id
    else:
        vid_path = vids[0]        
        
    tmp = glob(f'{sdir}/../preprocessing/{bn}*landmarks*')
    if len(tmp) == 0:
        continue
    
    lmks_path = tmp[0]
    
    # print(len(tmp))
    print('process_w3DID')
    print(vid_path)
    print(lmks_path)
    
    # pbn = os.path.basename('_'.join(vid_path.split('_')[:-1]))
    # # orig_vid_path = glob(f'/offline_data/face/CAR/InfantSync/id*/sess*/{pbn}*.mp4')[0]
    # orig_vid_path = glob(f'/offline_data/face/CAR/InfantSync/id*/sess*/{pbn}*.mp4')[0]
    
    vf.process_w3DID(vid_path, lmks_path)
    try:
        vf.process_w3DID(vid_path, lmks_path)
    except:
        print('skipping')

sys.exit(0)
#%%


for id in ids:
    bn = os.path.basename(id)
    
    vids = glob(f'/offline_data/face/CAR/InfantSync_3DI/preprocessing/{bn}*_2_und*mp4')
    
    for vid_path in vids:
        print(vid_path)
        lmks_path = vid_path.replace('.mp4', '.landmarks.global4')
        pbn = os.path.basename('_'.join(vid_path.split('_')[:-1]))
        orig_vid_path = glob(f'/offline_data/face/CAR/InfantSync/id*/sess*/{pbn}*.mp4')[0]
        try:
            vf.compute_pose_and_expression_coeffs(vid_path, lmks_path)
        except:
            print('Skipping')
            continue
        
#%%
ids = glob('/offline_data/face/CAR/InfantSync/id*')

for id in ids:
    bn = os.path.basename(id)
    
    vids = glob(f'/offline_data/face/CAR/InfantSync_3DI/preprocessing/{bn}*mp4')
    
    for vid_path in vids:
        print(vid_path)
        lmks_path = vid_path.replace('.mp4', '.landmarks.global4')
        pbn = os.path.basename('_'.join(vid_path.split('_')[:-1]))
        orig_vid_path = glob(f'/offline_data/face/CAR/InfantSync/id*/sess*/{pbn}*.mp4')[0]
        vf.visualize_video_output(vid_path, lmks_path, orig_vid_path=orig_vid_path)
        print('\n')
        
        
