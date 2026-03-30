from utils.graphics_utils import focal2fov, getWorld2View2, getWorld2View3
from PIL import Image
import torch
import numpy as np
from scene.gaussian_model import BasicPointCloud
from typing import NamedTuple
from lietorch import SE3

class CameraInfo(NamedTuple):
    uid: int
    tstamp: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: torch.Tensor
    
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View3(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def make_scene_info(config, video, tmp_pcd, cur_ix, downed_img):    
    h0 = config['Dataset']['Calibration']['height']
    w0 = config['Dataset']['Calibration']['width']
    
    fx = config['Dataset']['Calibration']['fx']
    fy = config['Dataset']['Calibration']['fy']
    cx = config['Dataset']['Calibration']['cx']
    cy = config['Dataset']['Calibration']['cy']
    
    if downed_img:
        h1, w1 = video.ref_image[0].shape[1:3]
        fx, cx = fx * (w1 / w0), cx * (w1 / w0)
        fy, cy = fy * (h1 / h0), cy * (h1 / h0)
        width, height = w1, h1
    else:
        
        width, height = w0, h0
    
        
    FovY = focal2fov(fy, height)
    FovX = focal2fov(fx, width)

    cam_infos = []
    for idx, tstamp in enumerate(video.tstamp[:cur_ix+1].int().tolist()):
        if downed_img:
            orig_img = video.downed_img[tstamp].squeeze()[[2,1,0], :, :]
        else:
            orig_img = video.original_img[tstamp].squeeze()[[2,1,0], :, :]
        img = Image.fromarray(orig_img.permute(1,2,0).numpy())

        cam_to_world = SE3(video.poses[idx]).matrix().cpu().numpy()
        R, T = cam_to_world[:3, :3], cam_to_world[:3, 3]
        uid = idx
        depth = video.disps_up[idx].clone()
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=img,
                            image_path=idx, image_name=idx, width=width, height=height, tstamp=tstamp, depth=depth)
        cam_infos.append(cam_info)

    cam_infos.sort(key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    pcd = BasicPointCloud(np.array(tmp_pcd.points), np.array(tmp_pcd.colors), np.zeros_like(np.array(tmp_pcd.points)))
    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=cam_infos,
                        test_cameras=[],
                        nerf_normalization=nerf_normalization,
                        ply_path=None)
    
    return scene_info

def scene_update(gaussian_train, ref_img, uid, pose, is_gba = False):
    if pose is not None:
        pose = pose.detach().cpu().numpy()
    gaussian_train.scene.extend_from_camera(gaussian_train.video, ref_img, uid, pose, is_gba)

def load_reference_frame(gaussian_train, ix):
    tstamp = int(gaussian_train.video.tstamp[ix].item())
    if gaussian_train.downed_img:
        gaussian_train.ref_img = gaussian_train.video.downed_img[tstamp].squeeze()[[2,1,0], :, :] / 255.0

    else:
        gaussian_train.ref_img = (gaussian_train.video.original_img[tstamp].squeeze()[[2, 1, 0], :, :] / 255.0)
        
    if torch.all(gaussian_train.ref_img == 0):
        print("ref_img is all zero!!!")
        
    gaussian_train.ref_depth = gaussian_train.video.disps_up[ix]
    gaussian_train.ref_pose = SE3(gaussian_train.video.poses[ix]).matrix()