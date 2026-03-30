import numpy as np
import torch
import lietorch
    
def get_pose(video, index, device):
    w2c = lietorch.SE3(video.poses[index].clone()).to(device) # Tw(droid)_to_c
    c2w = w2c.inv().matrix()  # [4, 4]
    return c2w

def get_depth_and_pose(video, index, device):
    # with self.get_lock():
    est_depth = video.disps_up[index].clone().to(device)  # [h, w]
    c2w = get_pose(video, index, device)
    return est_depth, None, c2w

def save_video(video, path:str, n_kf):
    poses = []
    depths = []
    timestamps = []
    # valid_depth_masks = []
    for i in range(n_kf):
        depth, _, pose = get_depth_and_pose(video, i,'cpu')
        timestamp = video.tstamp[i].cpu()
        poses.append(pose)
        depths.append(depth)
        timestamps.append(timestamp)
        # valid_depth_masks.append(depth_mask)
    poses = torch.stack(poses,dim=0).numpy()
    depths = torch.stack(depths,dim=0).numpy()
    timestamps = torch.stack(timestamps,dim=0).numpy() 
    # valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
    # np.savez(path,poses=poses,depths=depths,timestamps=timestamps,valid_depth_masks=valid_depth_masks)
    np.savez(path,poses=poses,depths=depths,timestamps=timestamps)
    
def load_poses(path, n_kf):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for i in range(n_kf):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        # c2w[:3, 1] *= -1
        # c2w[:3, 2] *= -1
        poses.append(c2w)
    return poses

def align_kf_traj(npz_path, pose_path, n_kf):
    offline_video = dict(np.load(npz_path))
    traj_ref = []
    traj_est = []
    video_traj = offline_video['poses']
    gt_pose = load_poses(pose_path, n_kf)
    video_timestamps = offline_video['timestamps']
    timestamps = []

    for i in range(video_timestamps.shape[0]):
        timestamp = int(video_timestamps[i])
        # val = stream.poses[timestamp].sum()
        # if np.isnan(val) or np.isinf(val):
            # printer.print(f'Nan or Inf found in gt poses, skipping {i}th pose!',FontColor.INFO)
            # continue
        traj_est.append(video_traj[i])
        traj_ref.append(gt_pose[i])
        timestamps.append(video_timestamps[i])

    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)

    # if return_full_est_traj:
    #     from evo.core import lie_algebra as lie
    #     traj_est_full = PoseTrajectory3D(poses_se3=video_traj,timestamps=video_timestamps)
    #     traj_est_full.scale(s)
    #     traj_est_full.transform(lie.se3(r_a, t_a))
    #     traj_est = traj_est_full

    # return r_a, t_a, s, traj_est, traj_ref   
    return s

