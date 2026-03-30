#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from PIL import Image
from droid_slam.scene.laplacian_model import LaplacianModel
from droid_slam.utils.camera_utils import cameraList_from_camInfos, loadCam
from droid_slam.utils.scene_utils import CameraInfo, getNerfppNorm
from lietorch import SE3
import torch
import numpy as np
from utils.graphics_utils import getWorld2View2, getWorld2View3, getProjectionMatrix

import open3d as o3d
from utils.sh_utils import eval_sh

class Scene:
    def __init__(self, scene_info, gaussians : LaplacianModel, kf_idx: int):

        self.gaussians = gaussians
        self.model_path = os.getcwd()
        self.train_cameras = {}

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # self.cameras_extent = 6.0

        resolution_scale = 1.0
        self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale)
        
        self.FoVx, self.FoVy =self.train_cameras[1.0][0].FoVx, self.train_cameras[1.0][0].FoVy
        
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, pcd_idx = kf_idx)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def get_TrainCamera_length(self, scale=1.0):
        return len(self.train_cameras[scale])
    
    def extend_from_camera(self, video, image, uid, pose, is_gba):
        # GBA일 때는 카메라만 업데이트
        if is_gba:
            for cam in self.train_cameras[1.0]:
                video_pose = SE3(video.poses[cam.uid]).matrix().detach().cpu().numpy()
                
                R = video_pose[:3, :3]
                T = video_pose[:3, 3]
            
                world_view_transform = torch.tensor(getWorld2View3(R, T)).transpose(0, 1).cuda()
                projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                
                cam.update_transform(R, T, world_view_transform, projection_matrix, full_proj_transform, camera_center)
        
        # GBA 아닐 때는 카메라 추가
        else :
            R = pose[:3, :3]
            T = pose[:3, 3]

            img = (image * 255).to(torch.uint8)
            img = Image.fromarray(img.permute(1,2,0).detach().cpu().numpy())
            
            depth = video.disps_up[uid].clone()
            
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=self.FoVy, FovX=self.FoVx, image=img,
                    image_path=uid, image_name=uid, width=image.shape[2], height=image.shape[1], tstamp = int(video.tstamp[uid].item()), depth=depth)
            extend_cam = loadCam(uid, cam_info, 1.0)
            
            self.train_cameras[1.0].append(extend_cam)

        self.cameras_extent = getNerfppNorm(self.train_cameras[1.0])['radius']

    def update_sampled_training(self, sampled_imgs, sampled_trajectory, sampled_tstamps):
        cam_infos = {}
        for (img, q_pose, tstamp) in zip(sampled_imgs, sampled_trajectory, sampled_tstamps):
            pose = SE3(q_pose).matrix()
            R = pose[:3, :3]
            T = pose[:3, 3]
            img = Image.fromarray(img[0].permute(1,2,0).detach().cpu().numpy())
            cam_infos[tstamp] = [R,T,img]
        return cam_infos
    
    def create_camlist(self, selected_infos):
        cam_infos = []
        for tstamp in selected_infos.keys():
            R = selected_infos[tstamp][0].detach().cpu().numpy()
            T = selected_infos[tstamp][1].detach().cpu().numpy()
            img = selected_infos[tstamp][2]
            cam_info = CameraInfo(uid=tstamp, R=R, T=T, FovY=self.FoVy, FovX=self.FoVx, image=img,
                image_path=tstamp, image_name=tstamp, width=img.size[0], height=img.size[1], tstamp = tstamp, depth=None)
            cam_infos.append(cam_info)
        extend_cam = cameraList_from_camInfos(cam_infos, 1)
        
        return extend_cam