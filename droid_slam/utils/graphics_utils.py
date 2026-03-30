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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

# def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
#     translate = translate.to(R.device)
#     Rt = torch.zeros((4, 4), device=R.device)
#     # Rt[:3, :3] = R.transpose()
#     Rt[:3, :3] = R
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0

#     C2W = torch.linalg.inv(Rt)
#     cam_center = C2W[:3, 3]
#     cam_center = (cam_center + translate) * scale
#     C2W[:3, 3] = cam_center
#     Rt = torch.linalg.inv(C2W)
#     return Rt


def getWorld2View3(R, t):
    # 그냥 그대로 출력 (4x4)
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros((4, 4), device="cuda:0")

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def batch_quaternion_to_rotation_matrix(quaternions):
    """
    Converts a batch of quaternions to a batch of 3x3 rotation matrices.
    quaternions: Tensor of shape (N, 4) where N is the number of quaternions
    Returns a Tensor of shape (N, 3, 3) representing the rotation matrices
    """
    # Extract the individual components from the quaternions
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    # Calculate the 3x3 rotation matrices for each quaternion in the batch
    rotation_matrices = torch.stack([
        torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)], dim=1),
        torch.stack([2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)], dim=1),
        torch.stack([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)], dim=1)
    ], dim=1)

    return rotation_matrices

def batch_rotation_matrix_to_quaternion(rotation_matrices, eps=1e-6):
    """
    Converts a batch of 3x3 rotation matrices to a batch of quaternions.
    rotation_matrices: Tensor of shape (N, 3, 3) where N is the number of rotation matrices
    Returns a Tensor of shape (N, 4) representing the quaternions (w, x, y, z)
    """
    R = rotation_matrices
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # Trace of the matrix
    
    # Handle different cases based on the trace value
    w = torch.sqrt(torch.clamp(1.0 + trace, min=eps)) / 2.0  # Ensure non-negative sqrt
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = torch.zeros_like(w)

    # Create mask for cases where w is near 0 to avoid divide by zero
    mask_w = w > eps

    # For cases where w is not too small, compute x, y, z normally
    x[mask_w] = (R[mask_w, 2, 1] - R[mask_w, 1, 2]) / (4.0 * w[mask_w])
    y[mask_w] = (R[mask_w, 0, 2] - R[mask_w, 2, 0]) / (4.0 * w[mask_w])
    z[mask_w] = (R[mask_w, 1, 0] - R[mask_w, 0, 1]) / (4.0 * w[mask_w])

    # Handle the cases where w is near zero separately
    # This could happen if trace is very small or negative (i.e., pure rotation)
    mask_x = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & ~mask_w
    mask_y = (R[:, 1, 1] > R[:, 2, 2]) & ~mask_w & ~mask_x
    mask_z = ~mask_w & ~mask_x & ~mask_y

    # If w is too small, calculate quaternion components using a different formula
    x[mask_x] = torch.sqrt(torch.clamp(1.0 + R[mask_x, 0, 0] - R[mask_x, 1, 1] - R[mask_x, 2, 2], min=eps)) / 2.0
    y[mask_x] = (R[mask_x, 0, 1] + R[mask_x, 1, 0]) / (4.0 * x[mask_x])
    z[mask_x] = (R[mask_x, 0, 2] + R[mask_x, 2, 0]) / (4.0 * x[mask_x])

    y[mask_y] = torch.sqrt(torch.clamp(1.0 + R[mask_y, 1, 1] - R[mask_y, 0, 0] - R[mask_y, 2, 2], min=eps)) / 2.0
    x[mask_y] = (R[mask_y, 0, 1] + R[mask_y, 1, 0]) / (4.0 * y[mask_y])
    z[mask_y] = (R[mask_y, 1, 2] + R[mask_y, 2, 1]) / (4.0 * y[mask_y])

    z[mask_z] = torch.sqrt(torch.clamp(1.0 + R[mask_z, 2, 2] - R[mask_z, 0, 0] - R[mask_z, 1, 1], min=eps)) / 2.0
    x[mask_z] = (R[mask_z, 0, 2] + R[mask_z, 2, 0]) / (4.0 * z[mask_z])
    y[mask_z] = (R[mask_z, 1, 2] + R[mask_z, 2, 1]) / (4.0 * z[mask_z])

    # Stack the components to form the quaternions
    quaternions = torch.stack([w, x, y, z], dim=1)

    return quaternions

def normalize_rotation_matrix(rotation_matrix):
    # rotation_matrix는 [N, 3, 3]의 형태
    u, _, v = torch.linalg.svd(rotation_matrix)  # 각 배치에 대해 SVD 수행
    rotation_matrix_normalized = torch.matmul(u, v.transpose(-2, -1))  # 배치마다 행렬 곱 수행
    return rotation_matrix_normalized


def normalize_quaternion(quaternion):
    # 쿼터니언 정규화
    norm = torch.norm(quaternion, dim=-1, keepdim=True)
    quaternion_normalized = quaternion / (norm + 1e-8)  # NaN을 방지하기 위해 작은 값 추가
    return quaternion_normalized
