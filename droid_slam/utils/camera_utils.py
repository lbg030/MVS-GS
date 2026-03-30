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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from utils.scene_utils import CameraInfo
import init_gaussian_train as GaussianTrain
import torch
from PIL import Image
import torch.nn.functional as F

def _build_camera(cam_info, include_depth):
    orig_w, orig_h = cam_info.image.size
    data_device = "cuda:0"
    resized_image_rgb = PILtoTorch(cam_info.image, (orig_w, orig_h))

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=cam_info.uid, data_device=data_device, tstmap=cam_info.tstamp, depth=cam_info.depth if include_depth else None)

def loadCam(id, cam_info, resolution_scale):
    return _build_camera(cam_info, include_depth=True)

def cameraList_from_camInfos(cam_infos, resolution_scale):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def ref_loadCam(id, cam_info, resolution_scale):
    return _build_camera(cam_info, include_depth=False)
    
def pyramid_cam(gaussian_train: GaussianTrain, cam, freq):
    if freq >= len(gaussian_train.pyramid_scale):
        return cam
    
    scale_factor = gaussian_train.pyramid_scale[freq]
    image = cam.original_image

    new_image = F.interpolate(image.unsqueeze(0), scale_factor=scale_factor, mode='bilinear').squeeze(0)
    new_img = (new_image * 255).to(torch.uint8)
    new_img = Image.fromarray(new_img.permute(1,2,0).detach().cpu().numpy())
    
    cam_info = CameraInfo(uid=cam.uid, R=cam.R, T=cam.T, FovY=cam.FoVy, FovX=cam.FoVx, image=new_img,
    image_path=cam.uid, image_name=cam.uid, width=new_image.shape[2], height=new_image.shape[1], tstamp = cam.tstamp, depth=cam.depth)
    extend_cam = loadCam(cam.uid, cam_info, 1.0)
    
    return extend_cam
