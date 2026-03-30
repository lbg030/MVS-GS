import os, cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image


def mvs_loader(config, tstamps, poses, ref_disp):
    h1 = config['tracking']['depth_fusion_size'][0]
    w1 = config['tracking']['depth_fusion_size'][1]
    
    transform = Compose([Resize((h1, w1)), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    
    k_fx = config['Dataset']['Calibration']['fx']
    k_fy = config['Dataset']['Calibration']['fy']
    k_cx = config['Dataset']['Calibration']['cx']
    k_cy = config['Dataset']['Calibration']['cy']
    
    fx = k_fx / 4
    fy = k_fy / 4
    cx = k_cx / 4
    cy = k_cy / 4
    
    k1 = config['Dataset']['Calibration']['k1']
    k2 = config['Dataset']['Calibration']['k2']
    p1 = config['Dataset']['Calibration']['p1']
    p2 = config['Dataset']['Calibration']['p2']
    k3 = config['Dataset']['Calibration']['k3']
    
    distortion_list = [k1, k2, p1, p2, k3]
    
    K = np.eye(3)
    K[0,0] = k_fx
    K[0,2] = k_cx
    K[1,1] = k_fy
    K[1,2] = k_cy
    images = []
    
    stride = config['Dataset']['stride']
    if config['Dataset']['scene_name'].startswith("scannet"):
        image_list = sorted(os.listdir(config['Dataset']['dataset_path']), key=lambda x: int(os.path.splitext(x)[0]))[::stride]
    else :
        image_list = sorted(os.listdir(config['Dataset']['dataset_path']))[::stride]
        
    h0, w0 = None, None
    for t in tstamps.cpu().numpy():
        imfile = image_list[int(t)]
        image = Image.open(os.path.join(config['Dataset']['dataset_path'], imfile))
        # if len(calib) > 4:
        #     image = cv2.undistort(np.array(image), K, calib[4:])
        if all(value != 0 for value in distortion_list):
            image = cv2.undistort(image, K, np.array(distortion_list))
            image = Image.fromarray(image)  # numpy.ndarray를 PIL.Image로 변환
        w0, h0 = image.size[:2]
        image = transform(image)
        images.append(image)
    images = torch.stack(images, dim=0)
    
    fx, cx = fx * (w1 / w0), cx * (w1 / w0)
    fy, cy = fy * (h1 / h0), cy * (h1 / h0)

    intr_mat = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=poses.device)
    
    intr_matrices = intr_mat.unsqueeze(0).repeat(poses.size(0), 1, 1)
    proj_matrices = torch.stack((poses, intr_matrices), dim=1)
    
    stage0_pjmats = proj_matrices.clone()
    stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
    stage1_pjmats = proj_matrices.clone()
    stage2_pjmats = proj_matrices.clone()
    stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
    stage3_pjmats = proj_matrices.clone()
    stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
    stage4_pjmats = proj_matrices.clone()
    stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

    proj_matrices = {"stage1": stage0_pjmats.unsqueeze(0),
                     "stage2": stage1_pjmats.unsqueeze(0),
                     "stage3": stage2_pjmats.unsqueeze(0),
                     "stage4": stage3_pjmats.unsqueeze(0),
                     "stage5": stage4_pjmats.unsqueeze(0)}
    
    ref_depth = 1 / ref_disp
    val_depths = ref_depth[(ref_depth > 0.01) & (ref_depth < 100)]
    min_d, max_d = torch.quantile(val_depths, torch.tensor([0.0, 0.90], device=val_depths.device)).cpu().numpy()
    d_interval = (max_d - min_d) / 192
    depth_values = torch.arange(0, 192, dtype=torch.float32).unsqueeze(0) * d_interval + min_d

    # return images.unsqueeze(0).cuda(), proj_matrices, depth_values.unsqueeze(0).cuda()
    return images.unsqueeze(0).cuda(), proj_matrices, depth_values.cuda()

def v_fuse_loader(args, tstamps, poses, ref_disp):
    h1, w1 = args.depth_fusion_size
    transform = Compose([Resize((h1, w1)), ToTensor()])
    calib = np.loadtxt(args.calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    k_fx, k_fy, k_cx, k_cy = calib[:4]
    
    K = np.eye(3)
    K[0,0] = k_fx
    K[0,2] = k_cx
    K[1,1] = k_fy
    K[1,2] = k_cy
    images = []
    image_list = sorted(os.listdir(args.imagedir))[::args.stride]
    
    for t in tstamps.cpu().numpy():
        imfile = image_list[int(t)]
        image = Image.open(os.path.join(args.imagedir, imfile))
        
        w0, h0 = image.size[:2]
        image = transform(image)
        images.append(image)
    images = torch.stack(images, dim=0)
    
    fx, cx = fx * (w1 / w0), cx * (w1 / w0)
    fy, cy = fy * (h1 / h0), cy * (h1 / h0)

    intr_mat = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=poses.device)
    
    intr_matrices = intr_mat.unsqueeze(0).repeat(poses.size(0), 1, 1)
    proj_matrices = torch.stack((poses, intr_matrices), dim=1).unsqueeze(0)
    
    ref_depth = 1 / ref_disp
    val_depths = ref_depth[(ref_depth > 0.01) & (ref_depth < 100)]
    min_d, max_d = torch.quantile(val_depths, torch.tensor([0.0, 0.90], device=val_depths.device)).cpu().numpy()
    d_interval = (max_d - min_d) / 192
    depth_values = torch.arange(0, 192, dtype=torch.float32).unsqueeze(0) * d_interval + min_d
    
    return images.unsqueeze(0).cuda(), proj_matrices, depth_values.cuda()