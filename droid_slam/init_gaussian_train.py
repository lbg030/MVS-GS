import numpy as np
import torch
import os
from random import randint
from gaussian_renderer import render_laplacian
from utils.multiprocessing_utils import clone_obj
from lietorch import SE3
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import LaplacianModel, Scene
from tqdm import tqdm
import time
from gui import gui_utils

from visualization_utils import *
from utils.loss_utils import l1_loss, ssim, smooth_l1_loss, depth_smoothness_loss, generate_psnr_mask 
from utils.scene_utils import make_scene_info, scene_update, load_reference_frame
from utils.camera_utils import  pyramid_cam
from utils.pcd_utils import create_point_actor, create_point_actor_from_indices, generate_pcd
from droid_slam.utils.nvs_utils import update_sampled_training, optimize_views
from utils.eval_utils import evaluation
from utils.traj_utils import save_video, align_kf_traj
from collections import OrderedDict

class GaussianTrain:
    def __init__(self, q_front2gs, q_main2vis, q_gs2main, video, config):
        # save parameters
        self.config = config
        self.video = video
        self.q_front2gs = q_front2gs
        self.q_main2vis = q_main2vis
        self.q_gs2main = q_gs2main

        # initialize parser
        parser = ArgumentParser(description="Training script parameters")
        self.dataset = ModelParams(parser)
        self.opt = OptimizationParams(parser, self.config)
        self.pipe = PipelineParams(parser, self.config) 
        self.set_hyperparams()    

    def set_hyperparams(self):
        # initialize training related variables
        self.global_iter = 0
        self.viewpoint_stack = None
        self.initialize = False
        self.scene = None
        self.eval = self.config['Visualize']['eval_rendering']
        self.device = self.config['model_params']['data_device']

        # set background color
        background_color = [1, 1, 1] if self.config['model_params']['white_background'] else [0, 0, 0]
        self.background = torch.tensor(background_color, dtype=torch.float32, device=self.device)

        # initialize Gaussian model
        self.gaussians = LaplacianModel(self.dataset.sh_degree)

        # set up progress bar for tracking progress
        self.max_iter = self.config['Visualize']['max_iter']  # maximum number of iterations
        self.progress_bar = tqdm(total=self.max_iter, desc="Training Progress")

        # initialize point cloud
        self.point_cloud = o3d.geometry.PointCloud()

        # set up voxel downsampling
        self.voxel_down = self.config['mapping']['voxel_down']
        self.voxel_size = self.config['mapping']['voxel_size']

        # set up PSNR
        self.psnr = self.config['mapping']['PSNR']['psnr']
        self.psnr_thresh = self.config['mapping']['PSNR']['psnr_thresh']

        # set up filters and thresholds
        self.filter_thresh = self.config['mapping']['Filtering']['filter_threshold']
        self.thresh_view = self.config['mapping']['Filtering']['thresh_view']
        self.thresh_disp = self.config['mapping']['Filtering']['thresh_disp']

        # initialize data structures
        self.tmp_point_cloud = {}
        self.key_frame_list = []
        self.old_depth_dic = {}
        self.old_pose_dic = {}  # world coordinates

        # set up pyramid
        self.pyramid = self.config['mapping']['Pyramid']['pyramid']
        self.pyramid_scale = self.config['mapping']['Pyramid']['pyramid_scale']
        self.pyramid_term = self.config['mapping']['Pyramid']['pyramid_term']
        
        self.downed_img = self.config['mapping']['downed_img']
        self.point_num_thresh = self.config['mapping']['point_num_thresh']

        # NVS
        self.top_k = self.config['mapping']['NVS']['top_k']
        self.frame_gap = self.config['mapping']['NVS']['frame_gap']
        
        # Visualize
        self.use_gui = self.config['Visualize']['use_gui']

        self.selected_views = {}
        self.just_views = {}
        
        self.selected_views_camera = None
        self.just_views_camera = None
        self.nvs_count = 0
        
    def run(self):
        while True:
            if not self.q_front2gs.empty():
                msg = self.q_front2gs.get()
                message_type = msg[0]

                if message_type == "init" or not self.initialize:
                    self.process_init_message(msg)
                elif message_type == "update":
                    self.process_update_message(msg)
                elif message_type == "GBA":
                    self.process_gba_message(msg)
                elif message_type == "break":
                    self.process_break_message(msg)
                    break  
            else:
                time.sleep(0.01)

            if self.initialize:
                self.train_gaussians()

        print("Training Done")

    def process_init_message(self, msg):
        if msg[0] != "init" and not self.initialize:
            dirty_ix_list, self.ix = msg[1], msg[2]
        else:
            dirty_ix_list, self.intr_mat, self.ix = msg[1], msg[2], msg[3]

        load_reference_frame(self, self.ix)
        self.width, self.height = self.ref_img.shape[2], self.ref_img.shape[1]

        tmp_idx_pcd_dic = {}
        for i in range(dirty_ix_list[0] + 1):
            ix_list = [i, i + 4]
            point_actor = create_point_actor_from_indices(self, ix_list)
            if point_actor is not None:
                self.point_cloud += point_actor
                tmp_idx_pcd_dic[ix_list[1] - 1] = point_actor
            self.old_pose_dic[ix_list[1] - 1] = SE3(self.video.poses[ix_list[1] - 1]).matrix().clone()

        completed = min((tmp_idx_pcd_dic.keys()))
        scene_info = make_scene_info(self.config, self.video, tmp_idx_pcd_dic[completed], completed, self.downed_img)
        self.gaussians = LaplacianModel(3)
        self.scene = Scene(scene_info, self.gaussians, completed)
        self.initialize = True
        self.gaussians.training_setup(self.opt)
        for key, value in tmp_idx_pcd_dic.items():
            if key != completed:
                self.gaussians.extend_from_pcd(value, self.scene.cameras_extent, pcd_idx=key)
        
        print(f"Initialization Done with {self.gaussians.get_xyz.shape[0]} points")

    def process_update_message(self, msg):
        dirty_ix_list, self.ix = msg[1], msg[2]

        load_reference_frame(self, self.ix)
        scene_update(self, self.ref_img, self.ix, self.ref_pose)

        self.old_pose_dic[self.ix] = self.ref_pose.clone()
        self.old_depth_dic[self.ix] = self.ref_depth.clone()

        view_cam = self.scene.getTrainCameras()[-1]
        render_pkg = render_laplacian(view_cam, self.gaussians, self.pipe, self.background)
        image = render_pkg["render"]
        render_depth = render_pkg["render_depth"]
        gt_image = view_cam.original_image.cuda()

        if self.psnr:
            l1_mask = generate_psnr_mask(image, gt_image, threshold=self.psnr_thresh)
            points, colors = generate_pcd(self, dirty_ix_list, l1_mask, rendered_depth = render_depth)
        else:
            points, colors = generate_pcd(self, dirty_ix_list)

        if points.shape[0] > self.point_num_thresh:
            point_actor = create_point_actor(self, points, colors)
            if len(point_actor.points) > self.point_num_thresh:
                self.gaussians.extend_from_pcd(point_actor, self.scene.cameras_extent, pcd_idx=self.ix)

    def process_gba_message(self, msg):
        sampled_imgs, sampled_trajectory, sampled_tstamps = msg[1], msg[2], msg[3]
        sampled_imgs = sampled_imgs[:, :, [2, 1, 0], ...]
        print("GS: GBA Processing")

        with torch.no_grad():
            self.old_pose_dic = self.gaussians.transform_gaussian(
                camera=self.scene.getTrainCameras().copy(),
                video=self.video,
                old_pose_dic=self.old_pose_dic,
            )
            scene_update(self, None, None, None, is_gba=True)
            
            new_training_view = update_sampled_training(self, sampled_imgs, sampled_trajectory, sampled_tstamps)
            
        print("GS: GBA Processing Done")
        print("GS: NVS Training")
        
        nvs_dic, selected_uids, just_frame_set = optimize_views(self, new_training_view)
        top_10_keys = sorted(nvs_dic, key=nvs_dic.get, reverse=True)[:20]
        
        self.gaussians.outlier_prune()
        uncertainty_values = self.gaussians.get_uncertainty
        threshold = torch.quantile(uncertainty_values, 0.99)
        mask = uncertainty_values >= threshold
        self.gaussians.densify_based_uncertainty(mask.squeeze())
        
        print("GS: NVS Training Done")
        self.q_gs2main.put(["done", top_10_keys])

    def process_break_message(self, msg):
        self.final_training_iters = msg[1]
    
        print("Final Iters")

        self.voxel_down = False
        
        for iter in range(self.final_training_iters):
            self.train_gaussians(final=True, final_iters = iter)
        
        # self.gaussians.calculate_coordinates(self.scene.getTrainCameras().copy(), self.video)
                
        from PIL import Image
        full_data = msg[2]
        sampled_imgs, sampled_trajs, sampled_tstamps = full_data[0], full_data[1], full_data[2]
        sampled_trajs = SE3(sampled_trajs).matrix()
        selected_infos = {tstamp:[traj[:3, :3], traj[:3, 3], Image.fromarray(img[0].permute(1,2,0).detach().cpu().numpy())] for (img, traj, tstamp) in zip(sampled_imgs, sampled_trajs, sampled_tstamps)}
        
        extend_cam = self.scene.create_camlist(selected_infos)
        
        # FIXME save gaussian splatting result
        # for cam in extend_cam:
        #     render_pkg = render_laplacian(cam, self.gaussians, self.pipe, self.background)
        #     render_image = render_pkg["render"]
        #     render_image = torch.clamp(render_image, 0.0, 1.0)
        #     render_depth = render_pkg["render_depth"]
        #     save_img(render_image, f"/home/lbg030/Desktop/gaussian_model/scenario3/using_data_images/render_{cam.tstamp}.png")
        #     save_depth(render_depth, f"/home/lbg030/Desktop/gaussian_model/scenario3/using_data_depth/depth_{cam.tstamp}.png")
            
        # self.gaussians.save_ply("/home/lbg030/Desktop/gaussian_model/scenario3/scenario3.ply")

        self.q_gs2main.put("final")
        if self.eval:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            global_scale = None
            # n_kf = len(self.viewpoint_stack)
            # save_path = self.config['model_params']['output_path']
            # pose_path = self.config['Dataset']['pose_path']
            with torch.no_grad():
                # save_video(self.video, path=f"/home/lbg030/Desktop/slam/MVS-Droid_SLAM/output/video.npz", n_kf=n_kf)
                # global_scale = align_kf_traj(f"/home/lbg030/Desktop/slam/MVS-Droid_SLAM/output/video.npz", pose_path, n_kf = n_kf)
                evaluation(self, global_scale=global_scale)
        self.q_gs2main.put("done")

    def train_gaussians(self, final=False, new_view=None, tracking_nvs_loss=False, final_iters = 0, just_frame=None):
        def process_view(v_cam, final, is_new_view=False):
            
            self.global_iter += 1
            self.progress_bar.update(1)
            
            if self.global_iter % 1000 == 0:
                self.gaussians.oneupSHdegree()
            
            self.gaussians.update_learning_rate(self.global_iter)
            
            render_pkg = render_laplacian(v_cam, self.gaussians, self.pipe, self.background)
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            render_depth = render_pkg["render_depth"]
            
            gt_image = v_cam.original_image.cuda()
            mask = gt_image > 0
            
            Ll1 = l1_loss(image, gt_image)
            d_smooth = depth_smoothness_loss(render_depth.squeeze())
            
            if not final:
                if not is_new_view and hasattr(v_cam, 'depth'):
                    mask = v_cam.depth != 0
                    depth_loss = smooth_l1_loss(render_depth.squeeze(), v_cam.depth.squeeze(), mask)
                    loss = (self.opt.labmda_l1 * Ll1 +
                            self.opt.lambda_dssim * (1.0 - ssim(image, gt_image)) +
                            self.opt.lambda_dssim * depth_loss)
                else:
                    loss = (self.opt.labmda_l1 * Ll1 +
                            self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                             + 0.1 * d_smooth)
            else:
                loss = (self.opt.labmda_l1 * Ll1 +
                            self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                             + 0.1 * d_smooth)
            loss.backward()
            
            with torch.no_grad():
                if self.global_iter < self.opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                if self.global_iter > self.opt.densify_from_iter and self.global_iter % 150 == 0:
                    size_threshold = 20
                    self.gaussians.densify_and_prune(
                        self.opt.densify_grad_threshold,
                        self.opt.prune_opacity_threshold,
                        # 0.7,
                        self.scene.cameras_extent,
                        size_threshold,
                    )
                
                if not final or self.config['Dataset']['type'] != "replica":
                    if self.global_iter > self.opt.densify_from_iter and self.global_iter % self.opt.shape_pruning_interval == 0:
                        self.gaussians.size_prune(self.opt.prune_shape_threshold)
                    if self.global_iter % self.opt.opacity_reset_interval == 0:
                        self.gaussians.reset_opacity()
                    if self.global_iter % self.opt.shape_reset_interval == 0:
                        self.gaussians.reset_shape()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                        
                return loss

        if new_view is not None:
            if tracking_nvs_loss:
                nvs_loss = {}
            while new_view:
                v_cam = new_view.pop(randint(0, len(new_view) - 1))
                loss = process_view(v_cam, final, is_new_view=True)
                if v_cam.depth is not None:
                    v_cam.freq += 1
                if tracking_nvs_loss:
                    nvs_loss[v_cam.tstamp] = loss.item()
            if tracking_nvs_loss:
                return nvs_loss
            
        else:
            if not final:
                if not self.viewpoint_stack:
                    self.viewpoint_stack = self.scene.getTrainCameras().copy()
                
                v_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
                v_cam.freq += 1
                
                if self.pyramid and not final:
                    v_cam = pyramid_cam(self, v_cam, int(v_cam.freq // self.pyramid_term))
                process_view(v_cam, final)
            
            else :
                if not self.viewpoint_stack:
                    self.viewpoint_stack = self.scene.getTrainCameras().copy()
                    
                    
                v_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
                v_cam.freq += 1
                # self.gaussians.remove_sky(v_cam)
                process_view(v_cam, final)
        
        with torch.no_grad():
            if self.use_gui:
                if self.global_iter % 10 == 0:
                    if not final:
                        self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                gaussians=clone_obj(self.gaussians),
                                current_frame=self.scene.getTrainCameras()[-1],
                                keyframes=self.scene.getTrainCameras().copy(),
                                gtcolor=self.ref_img.clone().detach().cpu(),
                                gtdepth=self.ref_depth.unsqueeze(0).clone().detach().cpu(),
                                selected_nvs = self.selected_views_camera if self.selected_views_camera is not None else None,
                                just_frame = self.just_views_camera if self.just_views_camera is not None else None,
                            )
                        )
                    else:
                        self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                gaussians=clone_obj(self.gaussians),
                                # keyframes=self.scene.getTrainCameras().copy(),
                                # gtcolor=self.ref_img.clone().detach().cpu(),
                                # gtdepth=self.ref_depth.unsqueeze(0).clone().detach().cpu(),
                                # selected_nvs = self.selected_views_camera,
                                # just_frame = self.just_views_camera,
                            )
                        )
                            
                torch.cuda.empty_cache()
        
def gauss_train_run(q_front2gs, q_main2vis, q_gs2main, video, config):
    try:
        gs = GaussianTrain(q_front2gs, q_main2vis, q_gs2main, video, config)
        gs.run()
    finally:
        q_front2gs.close()
        q_main2vis.close()
        q_gs2main.close()
        
        del gs
        torch.cuda.empty_cache()