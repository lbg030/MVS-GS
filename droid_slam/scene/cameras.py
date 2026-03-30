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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getWorld2View3, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, H=None, W=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 tstmap = 0, depth = None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        
        # 박준규 07/26
        self.freq = 0
        self.tstamp = tstmap
        self.depth = depth

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else :
            self.image_width = W
            self.image_height = H
            self.original_image = torch.zeros((1, self.image_height, self.image_width))
        
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        self.zfar = 100.0
        self.znear = 0.001

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View3(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        R = T[:3, :3]
        t = T[:3, 3]
        return Camera(
            colmap_id=uid,
            R=R,
            T=t,
            FoVx=FoVx,
            FoVy=FoVy,
            image=None,
            gt_alpha_mask = None,
            image_name=None,
            uid = uid,
            H=H,
            W=W,
        )
    
    def update_transform(self, R, T, world_view_transform, projection_matrix, full_proj_transform, camera_center):
        self.R = R
        self.T = T
        self.world_view_transform = world_view_transform
        self.projection_matrix = projection_matrix
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

