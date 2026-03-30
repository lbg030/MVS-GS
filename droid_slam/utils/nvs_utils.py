import init_gaussian_train as GaussianTrain
import torch
from tqdm import tqdm

def update_sampled_training(gaussian_train: GaussianTrain, sampled_imgs, sampled_trajectory, sampled_tstamps):
    new_training_view = gaussian_train.scene.update_sampled_training(sampled_imgs, sampled_trajectory, sampled_tstamps)
    return new_training_view

def mark_visible(gaussian_train, R, T):
    height, width = gaussian_train.config['tracking']['depth_fusion_size']

    points = gaussian_train.gaussians.get_xyz
    points_camera = torch.matmul(points, R.T) + T

    original_fx = gaussian_train.config['Dataset']['Calibration']['fx']
    original_fy = gaussian_train.config['Dataset']['Calibration']['fy']
    
    original_width = gaussian_train.config['Dataset']['Calibration']['width']
    original_height = gaussian_train.config['Dataset']['Calibration']['height']

    new_fx = original_fx * (width / original_width)
    new_fy = original_fy * (height / original_height)

    idx_in_view = (
        (points_camera[:, 2] > 0) &
        (torch.abs(points_camera[:, 0]) < points_camera[:, 2] * (width / (2 * new_fx))) &
        (torch.abs(points_camera[:, 1]) < points_camera[:, 2] * (height / (2 * new_fy)))
    )

    return idx_in_view

def random_sampling(gaussian_train: GaussianTrain, new_training_view):
    
    import random
    uid_list = list(new_training_view.keys())
    selected_uids = random.sample(uid_list, gaussian_train.top_k)
    
    selected_infos = {uid: new_training_view[uid] for uid in selected_uids}
    real_training_view = gaussian_train.scene.create_camlist(selected_infos)
    real_training_view += gaussian_train.scene.getTrainCameras().copy()[-30:]

    for i in tqdm(range(10), desc="Training Gaussians"):
        if i < 9:
            gaussian_train.train_gaussians(final=False, new_view=real_training_view.copy(), tracking_nvs_loss=False)
        else:
            nvs_loss_dic = gaussian_train.train_gaussians(final=False, new_view=real_training_view.copy(), tracking_nvs_loss=True)
    return nvs_loss_dic

def optimize_views(gaussian_train: GaussianTrain, new_training_view):
    all_gaussians_xyz = gaussian_train.gaussians.get_xyz  # (N, 3)
    
    cov3x3 = gaussian_train.gaussians.convert_cov6_to_cov3x3_torch()  # (N, 3, 3)
    gs_uncertainty = gaussian_train.gaussians.get_uncertainty.squeeze()
    
    uid_list = list(new_training_view.keys())

    R_list = [new_training_view[uid][0] for uid in new_training_view.keys()]
    T_list = [new_training_view[uid][1] for uid in new_training_view.keys()]
    
    R_tensor = torch.stack(R_list)  # (V, 3, 3)
    T_tensor = torch.stack(T_list)  # (V, 3)

    uncertainty_dict = {}

    for i, view_uid in enumerate(uid_list):
        R = R_tensor[i]  # (3, 3)
        T = T_tensor[i]  # (3,)

        visible_mask = mark_visible(gaussian_train, R, T)  # (N,), torch.BoolTensor

        if not visible_mask.any():
            continue

        visible_gaussians_xyz = all_gaussians_xyz[visible_mask]  # (M, 3)
        visible_cov3x3 = cov3x3[visible_mask]  # (M, 3, 3)

        mu_camera = torch.matmul(visible_gaussians_xyz, R.T) + T  # (M, 3)

        R_expanded = R.unsqueeze(0)  # (1, 3, 3)
        sigma_camera = torch.matmul(R_expanded, torch.matmul(visible_cov3x3, R_expanded.transpose(1, 2)))  # (M, 3, 3)

        try:
            eigenvalues = torch.linalg.eigvalsh(sigma_camera)  # (M, 3)
        except:
            print("eigenvalues error")

        base_uncertainties = eigenvalues[:, -1] 
        model_uncertainties = gs_uncertainty[visible_mask].squeeze()  # (M,)
        
        lambda_ = 0.2
        combined_uncertainties = lambda_ * model_uncertainties + (1 - lambda_) * base_uncertainties 
        
        depths = mu_camera[:, 2]  # (M,)
        epsilon = 1e-6
        weights = (1.0 / (depths + epsilon) ** 2)
        weights /= weights.sum()

        weighted_uncertainties = combined_uncertainties * weights  # (M,)
        weighted_uncertainties = model_uncertainties * weights  # (M,)

        total_uncertainty = weighted_uncertainties.sum().item()

        uncertainty_dict[view_uid] = total_uncertainty

    sorted_views = sorted(uncertainty_dict.items(), key=lambda x: x[1], reverse=True)

    selected_uids = []
    suppressed_uids = set()
    frame_gap = gaussian_train.frame_gap
    max_frames = gaussian_train.top_k

    for uid, _ in sorted_views:
        if uid in suppressed_uids:
            continue
        selected_uids.append(uid)
        suppressed_uids.update(range(uid - frame_gap, uid + frame_gap + 1))
        if len(selected_uids) >= max_frames:
            break

    selected_infos = {uid: new_training_view[uid] for uid in selected_uids}
    real_training_view = gaussian_train.scene.create_camlist(selected_infos)
    real_training_view += gaussian_train.scene.getTrainCameras().copy()[-30:]

    for i in tqdm(range(10), desc="Training Gaussians"):
        if i < 9:
            gaussian_train.train_gaussians(final=False, new_view=real_training_view.copy(), tracking_nvs_loss=False, just_frame=None)
        else:
            nvs_loss_dic = gaussian_train.train_gaussians(final=False, new_view=real_training_view.copy(), tracking_nvs_loss=True, just_frame=None)
    return nvs_loss_dic, None, None
    