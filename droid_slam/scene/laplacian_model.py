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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation , var_approx , var_generalized, rotation_matrix_to_quaternion, quaternion_multiply
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, getWorld2View3, fov2focal, getWorld2View2
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import eval_sh

from lietorch import SE3
import torch.nn.functional as F
from visualization_utils import *


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)

    return symm

class LaplacianModel:
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation =  inverse_sigmoid
        self.opacity_activation = torch.sigmoid
        self.shape_activation = var_approx

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._shape = torch.empty(0)
        self._opacity = torch.empty(0)
        self._uncertainty = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.shape_prune_threshold = 0
        self.spatial_lr_scale = 0
        self.shape_strngth = 1.0
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.alpha = 0.9  # EMA 파라미터
        self.running_min_grad = None
        self.running_max_grad = None

        self._idx = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._shape,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._shape,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        rot = self._rotation
        if rot.numel() == 0:
            return rot.new_empty((0, 4)) 
        if rot.dim() == 1:
            rot = rot.view(-1, 4)
        return F.normalize(rot, dim=1)
    
    @property
    def get_xyz(self):
        xyz = self._xyz
        return xyz.view(-1, 3)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_shape(self):
        return self.shape_activation(self._shape,self.shape_strngth)

    @property
    def get_idx(self):
        return self._idx
    
    @property
    def get_uncertainty(self):
        return self._uncertainty
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling*self.get_shape, scaling_modifier, self._rotation)

    def reset_uncertainty(self):
        self._uncertainty = torch.ones((self.get_xyz.shape[0], 1), device="cuda")
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities, shapes = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_final(
            fused_point_cloud, features, scales, rots, opacities, shapes, kf_id
        )

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info

        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["mapping"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth):
        downsample_factor = 32
        point_size = 0.05
        point_size = self.config["mapping"]["point_size"]

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View3(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # Compute point distances with knn
        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        scales = scales.repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        shapes = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        return fused_point_cloud, features, scales, rots, opacities, shapes
    
    def extend_from_final(self, fused_point_cloud, features, scales, rots, opacities, shapes, kf_id):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_shapes = nn.Parameter(shapes.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))
        
        new_idx = torch.full((new_xyz.shape[0],), kf_id, dtype=torch.int, device="cuda")
        new_coords = torch.full((new_xyz.shape[0], 2), 0, dtype=torch.int, device="cuda")
        new_uncertainty = torch.ones((new_xyz.shape[0], 1), device="cuda")
        
        self.densification_postfix(new_xyz = new_xyz,
                                new_features_dc = new_features_dc,
                                new_features_rest = new_features_rest,
                                new_opacities = new_opacity,
                                new_shapes = new_shapes,
                                new_scaling = new_scaling,
                                new_rotation = new_rotation,
                                new_idx = new_idx,
                                new_coords = new_coords,
                                new_uncertainty = new_uncertainty)
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, pcd_idx : int = 0):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        shapes = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._shape = nn.Parameter(shapes.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
        self._idx = torch.full((self.get_xyz.shape[0],), pcd_idx, dtype=torch.int, device="cuda")
        self._uncertainty = torch.ones((self.get_xyz.shape[0], 1), device="cuda")

    def create_pcd_from_pcd(self, pcd : BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        shapes = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        
        return fused_point_cloud, features, scales, rots, opacities, shapes
    
    def extend_from_pcd(self, pcd, spatial_lr_scale, pcd_idx: int = 0):
        self.spatial_lr_scale = spatial_lr_scale
        
        fused_point_cloud, features, scales, rots, opacities, shapes = self.create_pcd_from_pcd(pcd)
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_shapes = nn.Parameter(shapes.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))
        
        new_idx = torch.full((new_xyz.shape[0],), pcd_idx, dtype=torch.int, device="cuda")
        new_uncertainty = torch.ones((new_xyz.shape[0], 1), device="cuda")
        
        self.densification_postfix(new_xyz = new_xyz,
                                new_features_dc = new_features_dc,
                                new_features_rest = new_features_rest,
                                new_opacities = new_opacity,
                                new_shapes = new_shapes,
                                new_scaling = new_scaling,
                                new_rotation = new_rotation,
                                new_idx = new_idx,
                                new_uncertainty = new_uncertainty)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.prune_shape_threshold = training_args.prune_shape_threshold
        self.shape_strngth = training_args.shape_strngth
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._shape], 'lr': training_args.shape_lr, "name": "shape"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        
        scale = (self._scaling * self.get_shape).detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def reset_shape(self):
        shapes_new = torch.zeros((self.get_opacity.shape[0], 1), dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(shapes_new, "shape")
        self._shape = optimizable_tensors["shape"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        shapes = torch.zeros((xyz.shape[0], 1), dtype=torch.float, device="cuda") 
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._shape = nn.Parameter(torch.tensor(shapes, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def replace_tensor_to_optimizer_shape(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._shape = optimizable_tensors["shape"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self._idx = self._idx[valid_points_mask]
        self._uncertainty = self._uncertainty[valid_points_mask]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,new_shapes, new_scaling, new_rotation, new_idx, new_uncertainty = None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "shape": new_shapes,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._shape = optimizable_tensors["shape"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self._idx = torch.cat((self._idx, new_idx), dim=0)
        self._uncertainty = torch.cat((self._uncertainty, new_uncertainty), dim=0)
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_shape = self._shape[selected_pts_mask].repeat(N,1)
        
        new_idx = self._idx[selected_pts_mask].repeat(N)
        new_uncertainty = self._uncertainty[selected_pts_mask].repeat(N,1)
        
        assert new_xyz.shape[0] == new_idx.shape[0]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,new_shape, new_scaling, new_rotation, new_idx, new_uncertainty)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_shapes = self._shape[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        new_idx = self._idx[selected_pts_mask]
        new_uncertainty = self._uncertainty[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,new_shapes, new_scaling, new_rotation, new_idx, new_uncertainty)

    def densify_based_uncertainty(self, mask):
        new_xyz = self._xyz[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_opacities = self._opacity[mask]
        new_shapes = self._shape[mask]
        new_scaling = self._scaling[mask]
        new_rotation = self._rotation[mask]
        
        new_idx = self._idx[mask]
        new_uncertainty = self._uncertainty[mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,new_shapes, new_scaling, new_rotation, new_idx, new_uncertainty)
        
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads = (1+self._uncertainty) * grads
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # self.densify_mask(prune_mask)

        torch.cuda.empty_cache()
    
    def densify_mask(self, selected_pts_mask, N = 2):
        stds = self.get_scaling[selected_pts_mask].repeat(N,1) 
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_shape = self._shape[selected_pts_mask].repeat(N,1)
        
        new_idx = self._idx[selected_pts_mask].repeat(N)
        new_coords = self._coords[selected_pts_mask].repeat(N,1)
        new_uncertainty = self._uncertainty[selected_pts_mask].repeat(N,1)
        
        assert new_xyz.shape[0] == new_idx.shape[0]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,new_shape, new_scaling, new_rotation, new_idx, new_coords, new_uncertainty)
    
    def size_prune(self, min_shape):
        prune_mask = (self.get_shape < min_shape).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        
    def add_densification_stats(self, viewspace_point_tensor, update_filter, nvs_step=False):
        grads = viewspace_point_tensor.grad[update_filter, :2]
        grad_norm = torch.norm(grads, dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += grad_norm
        self.denom[update_filter] += 1

        if grad_norm.numel() == 0:
            print("grad_norm is empty")
            
        if grad_norm.numel() != 0:
            if self.running_min_grad is None or self.running_max_grad is None:
                current_min = grad_norm.min().item()
                current_max = grad_norm.max().item()
                self.running_min_grad = current_min
                self.running_max_grad = current_max
            else:
                current_min = grad_norm.min().item()
                current_max = grad_norm.max().item()
                self.running_min_grad = self.alpha * self.running_min_grad + (1 - self.alpha) * current_min
                self.running_max_grad = self.alpha * self.running_max_grad + (1 - self.alpha) * current_max
            
            diff = (self.running_max_grad - self.running_min_grad) + self.eps
            norm_scaled = (grad_norm - self.running_min_grad) / diff
            norm_scaled = torch.clamp(norm_scaled, 0.0, 1.0)

            uncertainty_update = self._uncertainty[update_filter]
            new_uncertainty = self.beta1 * uncertainty_update + (1 - self.beta1) * norm_scaled
            new_uncertainty = torch.clamp(new_uncertainty, 0.0, 1.0)
            self._uncertainty[update_filter] = new_uncertainty

    def select_idx_gaussian(self, idx):
        idx_mask = (self._idx == idx)
        return idx_mask
    
    def transform_gaussian(self, camera, video, old_pose_dic):
        for cam in camera:
            project_idx = cam.uid
            idx_mask = self.select_idx_gaussian(project_idx)
            if idx_mask.sum().item() == 0:
                continue
            
            means = self.get_xyz.detach()
            points = means[idx_mask].detach()
            
            ref_pose = old_pose_dic[project_idx]
            new_pose = SE3(video.poses[project_idx]).matrix()
            
            transformation = torch.linalg.inv(torch.linalg.inv(ref_pose) @ new_pose)
            pix_ones = torch.ones((points.size(0), 1), device=points.device)
            pts4 = torch.cat([points, pix_ones], dim=1)
            means[idx_mask] = (transformation @ pts4.T).T[:, :3]
            
            self._xyz = self.replace_tensor_to_optimizer(means, "xyz")["xyz"]
            
            rots = self.get_rotation.detach()
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[idx_mask] = quaternion_multiply(transformation.expand_as(rots[idx_mask]), rots[idx_mask])
            
            with torch.no_grad():
                self._rotation = self.replace_tensor_to_optimizer(rots, "rotation")["rotation"]
        
        return old_pose_dic
    
    def outlier_prune(self):
        dist2 = distCUDA2(self.get_xyz)
        threshold_mask = (dist2 > torch.quantile(dist2, 0.99))
        
        print("Outlier Prune num : ", threshold_mask.sum().item())
        self.prune_points(threshold_mask)

    def convert_cov6_to_cov3x3_torch(self):
        cov6 = self.get_covariance()
        N = cov6.shape[0]
        device = cov6.device
        dtype = cov6.dtype
        cov3x3 = torch.zeros((N, 3, 3), device=device, dtype=dtype)
        cov3x3[:, 0, 0] = cov6[:, 0]  # sigma_xx
        cov3x3[:, 0, 1] = cov6[:, 1]  # sigma_xy
        cov3x3[:, 0, 2] = cov6[:, 2]  # sigma_xz
        cov3x3[:, 1, 0] = cov6[:, 1]  # sigma_xy symmetry
        cov3x3[:, 1, 1] = cov6[:, 3]  # sigma_yy
        cov3x3[:, 1, 2] = cov6[:, 4]  # sigma_yz
        cov3x3[:, 2, 0] = cov6[:, 2]  # sigma_xz symmetry
        cov3x3[:, 2, 1] = cov6[:, 4]  # sigma_yz symmetry
        cov3x3[:, 2, 2] = cov6[:, 5]  # sigma_zz
        return cov3x3
    
    def calculate_coordinates(self, camera, video):
        from utils.nvs_utils import mark_visible
        for cam in camera:
            R,T = cam.R, cam.T
            R = torch.from_numpy(R).to(torch.float32).to("cuda")
            T = torch.from_numpy(T).to(torch.float32).to("cuda")
            
            w2c = torch.eye(4, device="cuda")
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            
            idx_mask = mark_visible(self, R, T)
            points = self.get_xyz[idx_mask]
            if points.shape[0] > 0:                
                color = self.get_uncertainty[idx_mask]
                color_max = torch.quantile(color, 0.8)
                color_min = torch.quantile(color, 0.2)
                normalized_color = (color - color_min) / (color_max - color_min)
                cc = normalized_color * 255
                
                img_width = video.disps_up[cam.uid].shape[1]
                img_height = video.disps_up[cam.uid].shape[0]
                
                fx,fy = fov2focal(cam.FoVx, img_width), fov2focal(cam.FoVy, img_height)
                cx,cy = img_width/2 - 0.5, img_height/2 - 0.5
                
                K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device="cuda")

                points_camera = world_to_camera(points, w2c[:3, :3], w2c[:3, 3])
                points_2d = project_to_image(points_camera, K)
                points_2d = torch.round(points_2d).int()
                
                image, point_to_pixel_map = create_image_with_mapping(points_2d, cc, img_width, img_height)
                image = image.mean(axis=2)
                
                # mask = (
                #     (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width) &
                #     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)
                # )
                
                # final_mask = idx_mask.clone()
                # final_mask[idx_mask] = ~mask
                
                # coords_mask = idx_mask.clone()
                # coords_mask[idx_mask] = mask
                
                # points_2d = points_2d[mask]
                # x_coords = points_2d[:, 0].long()
                # y_coords = points_2d[:, 1].long()
                
                # self._coords[coords_mask, 0] = y_coords.int()
                # self._coords[coords_mask, 1] = x_coords.int()
            
                # self.prune_points(final_mask)

def world_to_camera(points, R, T):
    points_camera = torch.matmul(R, points.T) + T.unsqueeze(1)
    return points_camera.T

def project_to_image(points_camera, K):
    points_2d = torch.matmul(K, points_camera[:, :3].T)
    points_2d /= points_camera[:, 2].unsqueeze(0) 
    return points_2d[:2].T

def create_image_with_mapping(points_2d, colors, img_width, img_height):
    image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    point_to_pixel_map = []
    
    points_2d_int = points_2d.long()
    
    for i, point in enumerate(points_2d_int):
        x, y = point[0].item(), point[1].item()
        
        if 0 <= x < img_width and 0 <= y < img_height:
            image[y, x] = colors[i]
            point_to_pixel_map.append((i, (x, y)))
    
    return image, point_to_pixel_map