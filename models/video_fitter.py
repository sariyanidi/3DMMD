#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 07:53:55 2023

@author: sariyanide
"""
import os
import cv2
from utils import utils
import torch
import random
import numpy as np
import pandas as pd
import morphable_model
from torchvision.transforms import Grayscale
# from models.morphable_model import MorphableModel
from camera import Camera
from orthograhic_fitter import OrthographicFitter
from perspective_fitter import PerspectiveFitter
from medium_model import MediumModel
from separated_model import SeparatedModelV3
# import matplotlib.pyplot as plt




class VideoFitter():
    
    def __init__(self, cam, outdir_root=None, device='cuda', Tmax=None):
        self.cam = cam
        self.device = device
        self.outdir_root = outdir_root
        self.rasterize_fov = 15.0
        self.rasterize_size = 224.0
        self.rasterize_cam = Camera(fov_x=self.rasterize_fov, fov_y=self.rasterize_fov,
                                    cx=self.rasterize_size/2.0, cy=self.rasterize_size/2.0)
        self.use_3DID_exp_model = True
        self.Nframes = 7
        self.Ntot_reconstructions = 30
        self.Tmax = Tmax
        self.use_exp_for_neutral = True
        
        self.models_loaded = False
        self.which_bfm = 'BFMmm-23660'
        self.which_pts = 'sampled'
        
        self.mm = morphable_model.MorphableModel(key=self.which_bfm, device=self.device)
        
        
        if outdir_root is not None:
            if not os.path.exists(self.outdir_root):
                os.mkdir(self.outdir_root)
            
            self.outdir_3DID = f'{self.outdir_root}/{self.get_key(is_final=False)}'
            self.outdir_final = f'{self.outdir_root}/{self.get_key(is_final=True)}'
            
            if not os.path.exists(self.outdir_3DID):
                os.mkdir(self.outdir_3DID)
            
            if not os.path.exists(self.outdir_final):
                os.mkdir(self.outdir_final)
            
    
    def get_key(self, is_final):
        
        key = f'{self.use_exp_for_neutral}{self.use_3DID_exp_model}'
        
        if is_final:
            key = f'f{self.cam.fov_x}-{self.which_pts}-{self.Nframes}-{self.Ntot_reconstructions}-{key}'
            
        return key
    
    
    def load_models(self):
        
        if self.models_loaded:
            return
        
        # dataset used to train models
        dbname = 'combined_celeb_ytfaces'
        checkpoint_dir = 'checkpoints'
        
        cfgid = 2
        Ntra = 139979
        lrate = 1e-5
        backbone = 'resnet50'
        tform_data = True

        init_path_id = f'{checkpoint_dir}/medium_model{self.rasterize_fov:.2f}{dbname}{backbone}{Ntra}{tform_data}{lrate}-{cfgid}-BFMmm-23660UNL_STORED.pth'

        
        checkpoint_id = torch.load(init_path_id)
        if not self.use_3DID_exp_model:
            self.model_id =  MediumModel(rasterize_fov=self.rasterize_fov, 
                                                       rasterize_size=self.rasterize_size, 
                                                    label_means=checkpoint_id['label_means'].to(self.device), 
                                                    label_stds=checkpoint_id['label_stds'].to(self.device),
                                                    which_bfm=self.which_bfm, which_backbone=backbone)
    
            self.model_id.load_state_dict(checkpoint_id['model_state'])
            self.model_id.to(self.device)
            self.model_id.eval()
        else:
            init_path_perframe = init_path_id
    
            spath = f'{checkpoint_dir}/sep_modelv3SP{self.rasterize_fov:.2f}{dbname}{backbone}{lrate}{cfgid}{tform_data}{Ntra}_V2.pth'
    
            checkpoint = torch.load(spath)
            self.model = SeparatedModelV3(rasterize_fov=self.rasterize_fov, rasterize_size=self.rasterize_size,
                                                    label_means=checkpoint_id['label_means'].to(self.device), 
                                                    label_stds=checkpoint_id['label_stds'].to(self.device),
                                                    init_path_id=init_path_id,
                                                    init_path_perframe=init_path_perframe,
                                                    which_backbone=backbone)
    
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device)
            self.model.eval()
        
        self.models_loaded = True
    
    
    def get_landmarks(self, lmkspath):
        
        if lmkspath.find('.csv') >= 0:
            csv = pd.read_csv(lmkspath)
            L = csv.values[:,1:]
        else:
            L = np.loadtxt(lmkspath)
        
        return L
    
    
    def process_w3DID(self, vpath, lmkspath, params3DIlite_path=None):
        
        self.load_models()
        
        L = self.get_landmarks(lmkspath)
        bn = '.'.join(os.path.basename(vpath).split('.')[:-1])
        
        # all_params_path = f'{self.outdir_3DID}/{bn}_3DID.npy'
        # print(all_params_path)
        
        """
        if os.path.exists(all_params_path):
            return False
            return np.load(all_params_path, allow_pickle=True).item()
        """
        
        cap = cv2.VideoCapture(vpath)
        
        transform_gray = Grayscale(num_output_channels=3)
    
        alphas = []
        exps = []
        angles = []
        betas = []
        taus = []
        invMs = []
        frame_idx = 0
        
        while(True):    
            print('\rProcessing frame %d/%d'%(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), end="")
            frame_idx += 1
            ret, frame = cap.read()
        
            if not ret:
                break 
            
            lmks51 = L[frame_idx-1,:].reshape(-1,2).astype(np.float32)
            
            if lmks51.shape[0] == 68:
                lmks51 = lmks51[17:,:]
            
            if lmks51[0,0] == 0:
                alphas.append(alphas[-1] if len(alphas) > 0 else np.nan*np.ones((1, self.mm.Kid)))
                betas.append(betas[-1] if len(betas) > 0 else np.nan*np.ones((1, self.mm.Ktex)))
                angles.append(angles[-1] if len(angles) > 0 else np.nan*np.ones((1,3)))
                taus.append(taus[-1] if len(taus) > 0 else np.nan*np.ones((1, 3)))
                exps.append(exps[-1] if len(exps) > 0 else np.nan*np.ones((1, self.mm.Kexp)))
                invMs.append(invMs[-1] if len(invMs) > 0 else np.nan*np.ones((2, 3)))
                continue
            
            cim = frame.astype(np.float32)/255.0
            M = utils.estimate_norm(lmks51, cim.shape[0], 1.5, [25,25])
            cim = utils.resize_n_crop_cv(cim, M, int(self.rasterize_size))
            invM = utils.estimate_inv_norm(lmks51, frame.shape[1], 1.5, [25,25])
            
            """
            lmks51_hom = np.concatenate((lmks51, np.ones((lmks51.shape[0], 1))), axis=1)
            lmks_new = (M @ lmks51_hom.T).T
            icim = utils.resize_n_crop_inv_cv(cim, invM, (frame.shape[1], frame.shape[0]))
            if True: #frame_idx == 9:
                print(M)
                print(invM)
                plt.figure(figsize=(50, 70))
                plt.imshow(frame)
                plt.plot(lmks51_hom[:,0], lmks51_hom[:,1])
                plt.show()
                plt.imshow(cim)
                plt.plot(lmks_new[:,0], lmks_new[:,1])
                plt.show()
                # print(frame)
                print(icim.shape)
                plt.figure(figsize=(50, 70))
                plt.imshow(icim)
                plt.title('hey')
                plt.show()
                # break
                
            """

            cim = np.transpose(cim, (2,0,1))
            cim = transform_gray(torch.from_numpy(cim)).unsqueeze(0)
            cim = cim.to(self.device)
            
            if not self.use_3DID_exp_model:
                y = self.model_id(cim)
                params = self.model_id.parse_params(y[0])
            else:
                y, alpha_un, beta_un, _ = self.model.forward(cim)
                params = self.model.parse_params(y, alpha_un, beta_un)
    
            """
            if True:
                pts = self.mm.project_to_2d(self.rasterize_cam, params['angles'], params['tau'], params['alpha'], params['exp']).detach().cpu().squeeze().numpy()
                pts[:,1] = self.rasterize_size-pts[:,1]
                print(pts.shape)
                pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
                pts = (invM @ pts.T).T
                plt.figure(figsize=(1.4*50, 1.4*70))
                plt.imshow(frame)
                plt.plot(pts[:,0], pts[:,1], '.')
                plt.savefig('c.jpg')
                break            
            """

            
            """

            (mask, _, rim), pr = self.model.backbone_id.render_image(params)
            mask = mask.detach().cpu().numpy()[0,0,:,:]
            cim0 = cim.detach().cpu().numpy()[0,0,:,:]
            rim = rim.detach().cpu().numpy()[0,0,:,:]
            
            rim[mask==0] = (cim0[mask==0])
            rim[mask==1] = (cim0[mask==1]+2*rim[mask==1])/3.0
            """
                    
            """
            if frame_idx % 2  == 0:
                p = self.mm.compute_face_shape(params['alpha'], params['exp'])
                p = p.detach().cpu().squeeze().numpy()
                
                # plt.clf()
                plt.figure(frame_idx, figsize=(30*1.5,10*1.5))
                
                    
                plt.subplot(141)
                plt.imshow(cim0)
                
                plt.subplot(142)
                plt.imshow(rim)
                
                plt.subplot(143)
                # plt.plot(p0[:,0], p0[:,1], '.')
                plt.plot(p[:,0], p[:,1], '.')
                plt.ylim((-90, 90))
                
                
                plt.subplot(144)
                # plt.plot(p0[:,2], p0[:,1], '.')
                plt.plot(p[:,2], p[:,1], '.')
                plt.ylim((-90, 90))
                plt.show()
                """
            
            alphas.append(params['alpha'].detach().cpu().numpy().astype(float))
            betas.append(params['beta'].detach().cpu().numpy().astype(float))
            angles.append(params['angles'].detach().cpu().numpy().astype(float))
            taus.append(params['tau'].detach().cpu().numpy().astype(float))
            exps.append(params['exp'].detach().cpu().numpy().astype(float))
            invMs.append(invM)
        
        params3DIlite = {'alphas': alphas,
                      'betas': betas,
                      'angles': angles,
                      'taus': taus,
                      'exps': exps,
                      'invMs': invMs}
        
        if params3DIlite_path is not None:
            np.save(params3DIlite_path, params3DIlite)
        
        return params3DIlite
    
    def save_txt_files(self, params3DIlite, alphas_path, betas_path, expressions_path, poses_path, illums_path):
        
        alpha = np.mean(params3DIlite['alphas'], axis=0).T
        beta = np.mean(params3DIlite['betas'], axis=0).T
        
        np.savetxt(alphas_path, alpha)
        np.savetxt(betas_path, beta)
        
        taus = np.concatenate(params3DIlite['taus'], axis=0)
        angles = np.concatenate(params3DIlite['angles'], axis=0)
        
        poses = np.concatenate((taus, angles), axis=1)
        
        # Default illums parameter (used only for video visualization)
        illums = np.tile([48.06574, 9.913327, 798.2065, 0.005], (poses.shape[0], 1))
        
        np.savetxt(expressions_path, np.concatenate(params3DIlite['exps'], axis=0))
        np.savetxt(poses_path, poses)
        np.savetxt(illums_path, illums)
        
        
    
    def compute_neutral_face(self, vid_path, lmks_path, params3DIlite_path):
        
        bn = '.'.join(os.path.basename(vid_path).split('.')[:-1])
        alpha_path = f'{self.outdir_final}/{bn}.alpha'
        
        if os.path.exists(alpha_path):
            return np.loadtxt(alpha_path)
        
        all_params = self.process_w3DID(vid_path, lmks_path)
        random.seed(1907)
        
        of = OrthographicFitter(self.mm)
        
        pfs_f = PerspectiveFitter(self.mm, self.cam, use_maha=True, which_pts='lmks', F=1)
        pfd_f = PerspectiveFitter(self.mm, self.cam, use_maha=True, which_pts=self.which_pts, F=1)
        
        T = len(all_params['taus'])
        alphas = []
        taus = []
        us = []
        
        all_pts = []
        
        for t in range(0, min(T, 9000), 60):
            print(f'\rt={t}/T', end='')
            alpha = all_params['alphas'][t].to(self.device)
            exp = all_params['exps'][t].to(self.device)
            tau = all_params['taus'][t].to(self.device)
            angles = all_params['angles'][t].to(self.device)
            invM = all_params['invMs'][t]
            if self.use_exp_for_neutral:
                pts = self.mm.project_to_2d(self.rasterize_cam, angles, tau, alpha, exp).detach().cpu().squeeze().numpy()
            else:
                pts = self.mm.project_to_2d(self.rasterize_cam, angles, tau, alpha, exp*0).detach().cpu().squeeze().numpy()
            
            pts[:,1] = self.rasterize_size-pts[:,1]
            # print(pts.shape)
            pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
            pts = (invM @ pts.T).T
            # plt.plot(pts[:,0], pts[:,1], '.')
            # plt.show()
            # break
            
            face_size = np.sqrt((pts[:,0].max()-pts[:,0].min())**2+(pts[:,1].max()-pts[:,1].min())**2)
            # pts[:,1] = 1920-pts[:,1]
            # pts[:,1] = 224-pts[:,1]
            """
            if t % 3 == 0:
                plt.figure(figsize=(14.4*5,  19.2*5))
                plt.plot(pts[:,0], -pts[:,1], '.')
                plt.xlim((0, 1440))
                plt.ylim((-1920, 0))
                plt.show()
            continue
            """
            lmks = torch.from_numpy(pts).to(self.device)[self.mm.li,:]
            # lmks[:,1] = 1920-lmks[:,1]
            of_fit_params = of.fit_orthographic_GN(lmks.cpu().numpy(), plotit=False)[0]
            u = torch.tensor(of_fit_params['u'])
            tau = of.to_projective(of_fit_params, self.cam.get_matrix(), lmks.cpu().float())
        
            x0 = torch.cat([0*torch.rand(pfs_f.num_components), torch.tensor(tau), torch.tensor(u)]).to(self.device)
            # x0 = pfd_f.fit_GN(x0.float(), [torch.from_numpy(pts).to(mm.device).float()], plotit=True, use_ineq = False)[0]
            # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = True)[0]
            # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = False)[0]
            # break
            # x0 = pfd_f.fit_GN(x0.float(), [pts], plotit=True, use_ineq = True)[0]
            x0, fit_out = pfd_f.fit_GN(x0.float(), [torch.from_numpy(pts).to(self.device).float()], 
                                       plotit=False, plotlast=False, use_ineq = True)
            if fit_out['RMSE']/face_size > 3e-4:
                continue
            alphas.append(x0[:pfs_f.num_components])
            taus.append(x0[pfs_f.num_components:pfs_f.num_components+3])
            us.append(x0[pfs_f.num_components+3:pfs_f.num_components+6])
            
            all_pts.append(torch.from_numpy(pts).to(self.device).float())
        
        
        all_alphas = []
        
        for nreconstruct in range(self.Ntot_reconstructions):
            print(f'\r Rec: {nreconstruct}/{self.Ntot_reconstructions}', end='')
            idx = np.arange(len(taus)).astype(int)
            random.shuffle(idx)
            idx = idx[:self.Nframes]
            calphas = [alphas[x] for x in idx]
            ctaus = [taus[x] for x in idx]
            cus = [us[x] for x in idx]
            call_pts = [all_pts[x] for x in idx]
            pfd = PerspectiveFitter(self.mm, self.cam, use_maha=True, 
                                    which_pts=self.which_pts, 
                                    F=self.Nframes)
            x0 = pfd.construct_initialization(calphas, ctaus, cus)
            x, fit_out = pfd.fit_GN(x0.float(), call_pts, plotit=False, plotlast=False, use_ineq = True)
            
            if fit_out['RMSE']/face_size > 3e-4:
                continue
            
            all_alphas.append(x[:pfd.num_components])
        
        alpha = torch.stack(all_alphas).mean(dim=0).cpu().numpy()
        np.savetxt(alpha_path, alpha)
        
        return alpha

    
        
    def compute_pose_and_expression_coeffs(self, vid_path, lmks_path, params3DIlite_path):
        
        bn = '.'.join(os.path.basename(vid_path).split('.')[:-1])
        
        poses_fpath = f'{self.outdir_final}/{bn}.poses'
        exps_fpath = f'{self.outdir_final}/{bn}.expressions'
        
        if os.path.exists(poses_fpath) and os.path.exists(exps_fpath):
            return np.loadtxt(poses_fpath), np.loadtxt(exps_fpath)
        
        all_params = self.process_w3DID(vid_path, lmks_path)
        alpha = self.compute_neutral_face(vid_path, lmks_path) 
        alpha = torch.from_numpy(alpha).to(self.device).float()
        
        mm_alpha = morphable_model.MorphableModel(key=self.which_bfm, device=self.device)
        mm_alpha.update_mean_face(alpha)
        
        of = OrthographicFitter(mm_alpha)
        
        pfs = PerspectiveFitter(mm_alpha, self.cam, use_maha=False, which_pts='lmks', F=1, which_basis='expression')
        pfd = PerspectiveFitter(mm_alpha, self.cam, use_maha=False, which_pts=self.which_pts, F=1, which_basis='expression')
        
        us = []
        taus = []
        exps = []
        poses = []
        exp_prev = None
        u_prev = None
        tau_prev = None
        all_params['angles'] = all_params['angles'][::4]
        all_params['alphas'] = all_params['alphas'][::4]
        all_params['exps'] = all_params['exps'][::4]
        all_params['taus'] = all_params['taus'][::4]
        all_params['invMs'] = all_params['invMs'][::4]
        
        T = len(all_params['angles'])
        
        if self.Tmax is not None:
            T = min(self.Tmax, T)
        
        for t in range(T):
            print('\rProcessing frame %d/%d' % (t, T), end="")
        
            angles = all_params['angles'][t].to(self.device)
            tau = all_params['taus'][t].to(self.device)
            alpha = all_params['alphas'][t].to(self.device)
            exp = all_params['exps'][t].to(self.device)
            pts = self.mm.project_to_2d(self.rasterize_cam, angles, tau, alpha, exp).cpu().squeeze().to(self.device)
            pts[:,1] = self.rasterize_size-pts[:,1]
            pts = pts.cpu().numpy()
            pts = np.concatenate((pts, torch.ones((pts.shape[0], 1))), axis=1)
            invM = all_params['invMs'][t]
            
            pts = (invM @ pts.T).T
            
            lmks = pts[mm_alpha.li.cpu(),:]# .cpu().numpy()
            
            #if len(exps) == 0:
            of_fit_params = of.fit_orthographic_GN(lmks, plotit=False)[0]
            u0 = torch.tensor(of_fit_params['u']).to(self.device).flatten()
            tau0 = of.to_projective(of_fit_params, self.cam.get_matrix(), lmks)
            tau0 = torch.tensor(tau0).to(self.device).flatten()
            if t % 30 == 0:
                alpha0 = 0*torch.rand(pfd.num_components).to(self.device)
            else:
                alpha0 = x0[:pfd.num_components].to(self.device)
                

            x0 = torch.cat([alpha0, tau0, u0])
            
            # plotlast = t % 30 == 0
            x0 = pfd.fit_GN(x0.float(), [torch.from_numpy(pts).to(pfd.mm.device).float()], plotit=False, 
                            use_ineq = True, plotlast=False)[0]
            # print(time()-t1)
            exps.append(x0[:pfs.num_components].cpu().numpy())
            tau = x0[pfs.num_components:pfs.num_components+3].cpu().numpy()
            u = x0[pfs.num_components+3:pfs.num_components+6].cpu().numpy()
            
            taus.append(tau)
            us.append(u)
            
            pose = np.concatenate((tau, u), axis=0)
            poses.append(pose)
            tau0 = torch.from_numpy(tau).to(self.device).flatten()
            u0 = torch.from_numpy(u).to(self.device).flatten()
            
        poses = np.array(poses)
        exps = np.array(exps)
    
        np.savetxt(poses_fpath, poses) 
        np.savetxt(exps_fpath, exps)
        
        return poses, exps
    



    def visualize_video_output(self, vid_path, lmks_path, params3DIlite_path, orig_vid_path=None):
        
        bn = '.'.join(os.path.basename(vid_path).split('.')[:-1])
        
        if orig_vid_path is None:
            orig_vid_path = vid_path
        
        alpha = self.compute_neutral_face(vid_path, lmks_path)
        alpha = torch.from_numpy(alpha).to(self.device).float()
        poses, exps = self.compute_pose_and_expression_coeffs(vid_path, lmks_path, params3DIlite_path)
        
        T = exps.shape[0]
        # taus[1] *= -1
        # us[0] *= -1
        # us[-1] *= -1
        
        tex = np.loadtxt('/home/sariyanide/code/3DI/build/models/MMs/BFMmm-23660/tex_mu.dat').reshape(-1,1)
        illums = np.tile([48.06574, 9.913327, 798.2065, 0.005], (T, 1))

        # poses = np.zeros(poses.shape)
        # poses_np[:,3] = 0.
        # poses_np[:,4] = 0.11
        poses[:,2] -= 300
        # poses_np[:,3] *= -1
        # poses_np[:,3] -= np.pi
        poses[:,-1] *= -1
        # poses[:,3] *= -1
        # poses[:,4] *= -1
        # poses[:,-1] *= -1
        
        p0 = self.mm.compute_face_shape(alpha.unsqueeze(0))
        p0 = p0.detach().squeeze()
        p0[:,1] *= -1
        # p0[:,2] += 20
        # p0[:,2] = max(p0[:,2])-p0[:,2]
        
        shpsm_fpath = f'{self.outdir_final}/{bn}.shapesm'
        tex_fpath = f'{self.outdir_final}/{bn}.betas'
        illums_fpath = f'{self.outdir_final}/{bn}.illums'
        poses_fpath = f'{self.outdir_final}/{bn}.poses'
        
        exps_fpath = f'{self.outdir_final}/{bn}.expressions'
        cfg_fpath = f'/home/sariyanide/car-vision/cuda/build-3DI/configs/{self.which_bfm}.cfg1.global4.txt'
        render3ds_path = f'{self.outdir_final}/{bn}.avi' 
        texturefs_path = f'{self.outdir_final}/{bn}_texture_sm.avi' 
        
        np.savetxt(shpsm_fpath, p0.cpu().numpy())
        np.savetxt(tex_fpath, tex)
        np.savetxt(illums_fpath, illums)
        
        cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % (orig_vid_path, cfg_fpath, self.cam.fov_x, shpsm_fpath, tex_fpath,
                                                                           exps_fpath, poses_fpath, illums_fpath, 
                                                                           render3ds_path, texturefs_path)
        
        print('\n')
        print(cmd_vis)
        os.chdir('/home/sariyanide/car-vision/cuda/build-3DI')
        os.system(cmd_vis)
            

