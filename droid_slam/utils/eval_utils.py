import os
import torch
from tqdm import tqdm
import numpy as np
from visualization_utils import save_img, save_depth
from gaussian_renderer import render_laplacian
import torch.nn.functional as F

def evaluation(gaussian_train, global_scale=None):
    from init_gaussian_train import GaussianTrain  # Import inside function
    from gaussian_renderer import render_laplacian  # Import inside function

    os.makedirs("individual_results", exist_ok=True)
    with open(f"individual_results/{gaussian_train.config['Dataset']['scene_name']}_results.txt", "w") as f:
        f.write("Results\n")
    
    if not os.path.exists(f"/home/lsw/Desktop/psnr_mask/{gaussian_train.config['Dataset']['scene_name']}"):
        os.makedirs(f"/home/lsw/Desktop/psnr_mask/{gaussian_train.config['Dataset']['scene_name']}")
    
    if not os.path.exists(f"/home/lsw/Desktop/depth/{gaussian_train.config['Dataset']['scene_name']}"):
        os.makedirs(f"/home/lsw/Desktop/depth/{gaussian_train.config['Dataset']['scene_name']}")
    
    #MVS-GS
    # if not os.path.exists(f"/home/lbg030/Desktop/mvs-gs/{gaussian_train.config['Dataset']['scene_name']}/images"):
    #     os.makedirs(f"/home/lbg030/Desktop/mvs-gs/{gaussian_train.config['Dataset']['scene_name']}/images")
    
    # if not os.path.exists(f"/home/lbg030/Desktop/mvs-gs/{gaussian_train.config['Dataset']['scene_name']}/depths"):
    #     os.makedirs(f"/home/lbg030/Desktop/mvs-gs/{gaussian_train.config['Dataset']['scene_name']}/depths")
    
    def eval(
        pred_image,
        gt_image,
        pred_depth,
        global_scale,
        gt_depth = None
    ):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from droid_slam.utils.image_utils import psnr
        from droid_slam.utils.loss_utils import ssim
        
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")
        
        image_mask = gt_image > 0    
        
        psnr_score = psnr((pred_image[image_mask]).unsqueeze(0), (gt_image[image_mask]).unsqueeze(0))
        ssim_score = ssim((pred_image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((pred_image).unsqueeze(0), (gt_image).unsqueeze(0))
        
        # depth normalization
        # pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
        # gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())

        # valid_mask = (pred_depth > 0) * (gt_depth > 0)
        # edge 부분에 대한 mask 생성
        # edge_mask = create_edge_mask(pred_depth, edge_size=10)
        # mask = torch.logical_and(valid_mask, edge_mask)
        # mask = torch.logical_and(mask, pred_depth < 4.0)
        if gt_depth is not None:
            mask = (gt_depth) > 0
            mask = torch.logical_and(mask, gt_depth < 4.0)
            
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(pred_depth.unsqueeze(0), gt_depth.unsqueeze(0), mask.unsqueeze(0))
                pred_depth = scale * pred_depth + shift
            
            else :
                pred_depth = global_scale * pred_depth
            
            diff_depth_l1 = torch.abs(pred_depth[mask] - gt_depth[mask])
            depth_l1_gt = diff_depth_l1.sum() / mask.sum()
        else :
            depth_l1_gt = torch.tensor([1.0])
        # save_depth(gt_depth, f"/home/lbg030/Desktop/depth/{gaussian_train.config['Dataset']['scene_name']}/{cam.tstamp}_gt.png")
        # save_depth(pred_depth, f"/home/lbg030/Desktop/depth/{gaussian_train.config['Dataset']['scene_name']}/{cam.tstamp}_pred.png")
        
        return psnr_score.item(), ssim_score.item(), lpips_score.item(), depth_l1_gt.item()
    
    viewpoint_stack = gaussian_train.scene.getTrainCameras().copy()
    
    psnr_array, ssim_array, lpips_array, depthL1_array = [], [], [], []
    for cam in tqdm(viewpoint_stack):
        # render_laplacian 함수 호출 전                                              에 지연 import 수행
        render_pkg = render_laplacian(cam, gaussian_train.gaussians, gaussian_train.pipe, gaussian_train.background)
        image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = torch.clamp(image, 0.0, 1.0)
        
        depth = gaussian_train.video.disps_up[cam.uid].clone().detach().cpu()
        rendered_pkg = render_laplacian(cam, gaussian_train.gaussians, gaussian_train.pipe, gaussian_train.background)
        rendered_depth = rendered_pkg["render_depth"].squeeze(0).detach().cpu()
        try :
            gt_depth = load_depth_image(gaussian_train.config['Dataset']['depth_path'] + f"/depth{str(cam.tstamp).zfill(6)}.png", depth_size=(depth.shape[0], depth.shape[1]), depth_scale=gaussian_train.config['Dataset']['depth_scale'])
        except:
            gt_depth = None
        
        gt_image = cam.original_image.cuda()
        # psnr_score, ssim_score, lpips_score, mean_depthl1 = eval(image, gt_image, depth, gt_depth, global_scale)
        # psnr_score, ssim_score, lpips_score, mean_depthl1 = eval(image, gt_image, rendered_depth, gt_depth, global_scale)
        psnr_score, ssim_score, lpips_score, mean_depthl1 = eval(image, gt_image, rendered_depth, global_scale, gt_depth)

        # save_img(gt_image, f"/home/lbg030/Desktop/mvs-gs/{gaussian_train.config['Dataset']['scene_name']}/images/{cam.tstamp}_gt.png")
        # save_img(image, f"/home/lbg030/Desktop/mvs-gs/{gaussian_train.config['Dataset']['scene_name']}/images/{cam.tstamp}_rendered.png")
        
        # save_img(gt_image, f"/home/lbg030/Desktop/psnr_mask/{gaussian_train.config['Dataset']['scene_name']}/{cam.tstamp}_gt.png")
        # save_img(image, f"/home/lbg030/Desktop/psnr_mask/{gaussian_train.config['Dataset']['scene_name']}/{cam.tstamp}_rendered.png")
        
        # save_img(gt_image, f"/home/lbg030/Desktop/test/{cam.tstamp}_gt.png")
        # save_img(image, f"/home/lbg030/Desktop/test/{cam.tstamp}_rendered.png")
        
        with open(f"individual_results/{gaussian_train.config['Dataset']['scene_name']}_results.txt", "a") as f:
            f.write(f"tstamp : {cam.tstamp}: ")
            f.write(f"psnr : {psnr_score} ")
            f.write(f"ssim : {ssim_score} ")
            f.write(f"lpips : {lpips_score}\n")
            f.write(f"mean_depthL1 : {mean_depthl1}")
    
        psnr_array.append(psnr_score)
        ssim_array.append(ssim_score)
        lpips_array.append(lpips_score)
        depthL1_array.append(mean_depthl1)
    
    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_depthL1"] = float(np.mean(depthL1_array))
    
    print("psnr : " , output["mean_psnr"])
    print("ssim : " , output["mean_ssim"])
    print("lpips : " , output["mean_lpips"])
    print("mean_depthL1 : " , output["mean_depthL1"])
    
    total_num_gs = gaussian_train.gaussians.get_xyz.shape[0]
    with open("results.txt", "a") as f:
        f.write(f"dataset: {gaussian_train.config['Dataset']['scene_name']} voxel_size: {gaussian_train.voxel_size} Mask Thres: {gaussian_train.filter_thresh} \n")
        f.write(f"psnr : {output['mean_psnr']}\n")
        f.write(f"ssim : {output['mean_ssim']}\n")
        f.write(f"lpips : {output['mean_lpips']}\n")
        f.write(f"mean_depthL1 : {output['mean_depthL1']}\n")
        f.write(f"total_num_gs : {total_num_gs}\n")

@torch.no_grad()
def align_scale_and_shift(prediction, target, weights):

    '''
    weighted least squares problem to solve scale and shift: 
        min sum{ 
                  weight[i,j] * 
                  (prediction[i,j] * scale + shift - target[i,j])^2 
               }

    prediction: [B,H,W]
    target: [B,H,W]
    weights: [B,H,W]
    '''

    if weights is None:
        weights = torch.ones_like(prediction).to(prediction.device)
    if len(prediction.shape)<3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        weights = weights.unsqueeze(0)  
    a_00 = torch.sum(weights * prediction * prediction, dim=[1,2])
    a_01 = torch.sum(weights * prediction, dim=[1,2])
    a_11 = torch.sum(weights, dim=[1,2])
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(weights * prediction * target, dim=[1,2])
    b_1 = torch.sum(weights * target, dim=[1,2])
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b            
    det = a_00 * a_11 - a_01 * a_01
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    error = (scale[:,None,None]*prediction+shift[:,None,None]-target).abs()
    masked_error = error*weights
    error_sum = masked_error.sum(dim=[1,2])
    error_num = weights.sum(dim=[1,2])
    avg_error = error_sum/error_num

    return scale,shift,avg_error

def load_depth_image(depth_image_path, depth_size, depth_scale, device ="cpu"):
    import cv2
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
    depth_image = torch.tensor(depth_image, dtype=torch.float32, device=device)
    interpolated_depth = F.interpolate(depth_image.unsqueeze(0).unsqueeze(0), depth_size, mode="nearest").squeeze()
    return interpolated_depth

def create_edge_mask(pred_depth, edge_size = 10):
    H = pred_depth.shape[0]
    W = pred_depth.shape[1]
    
    mask = torch.ones((H,W), dtype=torch.bool, device=pred_depth.device)
    
    mask[:, :edge_size] = False
    mask[:, -edge_size:] = False
    
    mask[:edge_size, :] = False
    mask[-edge_size:, :] = False
    
    return mask