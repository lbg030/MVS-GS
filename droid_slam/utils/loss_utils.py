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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def compute_psnr(image, gt_image):
    mse = F.mse_loss(image, gt_image, reduction='none').mean(dim=0)  # H x W
    psnr = 10 * torch.log10(1 / mse)
    return psnr

def generate_psnr_mask(image, gt_image, threshold):
    with torch.no_grad():
        psnr_map = compute_psnr(image, gt_image)
        mask = psnr_map < threshold
    return mask

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def smooth_l1_loss(pred, target, mask, beta=1.0, reduction='mean'):
    """
    Smooth L1 Loss 함수 구현

    Args:
        pred (torch.Tensor): 예측 값
        target (torch.Tensor): 실제 값
        beta (float, optional): 전환점(transition point). 기본값은 1.0
        reduction (str, optional): 반환되는 손실 값의 형태. 'mean', 'sum', 'none' 중 하나 선택

    Returns:
        torch.Tensor: 계산된 손실 값
    """
    pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=target.shape[-2:], mode='bilinear').squeeze()
    pred = pred[mask]
    target = target[mask]
    diff = pred - target
    abs_diff = torch.abs(diff)
    # beta 값이 매우 작은 경우에는 L1 Loss로 근사
    if beta < 1e-5:
        loss = abs_diff
    else:
        loss = torch.where(
            abs_diff < beta,
            0.5 * (diff ** 2) / beta,
            abs_diff - 0.5 * beta
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def depth_smoothness_loss(depth_map):
    """
    인접 픽셀 간 depth 값의 차이를 최소화하는 smoothing loss

    Args:
        depth_map (torch.Tensor): (H, W) 형태의 Depth Map Tensor. GPU에 있어야 함.
        
    Returns:
        torch.Tensor: Smoothness loss 값 (scalar)
    """
    # x 방향 차이 계산: 오른쪽 픽셀과의 차이
    depth_dx = torch.abs(depth_map[:, :-1] - depth_map[:, 1:])
    
    # y 방향 차이 계산: 아래쪽 픽셀과의 차이
    depth_dy = torch.abs(depth_map[:-1, :] - depth_map[1:, :])
    
    # 전체 loss는 x, y 방향 차이의 평균
    loss = torch.mean(depth_dx) + torch.mean(depth_dy)
    
    return loss
