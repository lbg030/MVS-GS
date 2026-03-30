import torch
import torch.nn.functional as F
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops
import depth_fusion

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]]) / 3

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def down_sample(point_actor):
    def estimate_radius_and_neighbors(pcd, k=20):
        distances = pcd.compute_nearest_neighbor_distance()
        mean_distance = np.mean(distances)
        
        radius = 10 * mean_distance
        
        neighbors = int(k)
        
        return radius, neighbors
    voxel_size = 0.003
    voxel_down_pcd = point_actor.voxel_down_sample(voxel_size=voxel_size)
    radius, min_neighbors = estimate_radius_and_neighbors(voxel_down_pcd, k=20)

    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    filtered_pcd = voxel_down_pcd.select_by_index(ind)
    
    return filtered_pcd
    

def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.005):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def droid_visualization(args, video, in_queue, out_queue,device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.poses = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 0.01
    droid_visualization.thresh_view = 3
    droid_visualization.thresh_disp = 1.0
    
    calib = args['Dataset']['Calibration']
    fx,fy,cx,cy = calib['fx'], calib['fy'], calib['cx'], calib['cy']
    
    w0, h0 = args['Dataset']['Calibration']['width'], args['Dataset']['Calibration']['height']
    w1, h1 = args['tracking']['depth_fusion_size'][1], args['tracking']['depth_fusion_size'][0]
    
    intr_mat = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0],
                        [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=device)

    scale_w = 1.0 * w1 / w0
    scale_h = 1.0 * h1 / h0
    
    intr_mat[0, :] *= scale_w
    intr_mat[1, :] *= scale_h
    
    def animation_callback(vis):
        while True:
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            try:
                msg = in_queue.get_nowait()
                if msg == "Extract":    
                    print("Extracting...")
                    point_dic = {k: np.array(v.points) for k, v in droid_visualization.points.items()}
                    color_dic = {k: np.array(v.colors) for k, v in droid_visualization.points.items()}
                    out_queue.put(("Extract", point_dic, color_dic, droid_visualization.poses))
            except:
                with torch.no_grad():

                    with video.get_lock():
                        t = video.counter.value 
                        dirty_index, = torch.where(video.dirty.clone())
                        dirty_index = dirty_index
                    if (len(dirty_index) == 0) or (intr_mat is None):
                        return
                    video.dirty[dirty_index] = False
                    # convert poses to 4x4 matrix
                    poses = torch.index_select(video.poses, 0, dirty_index)
                    poses = SE3(poses).matrix() # [N, 4, 4]

                    depths = torch.index_select(video.disps_up, 0, dirty_index).unsqueeze(1).unsqueeze(0)
                    intr_matrices = intr_mat.unsqueeze(0).repeat(poses.size(0), 1, 1)
                    proj_matrices = torch.stack((poses, intr_matrices), dim=1).unsqueeze(0) # [N, 2, 4, 4]
                    ref_depth, src_depths = depths[:, -1, ...], depths[:, :-1, ...]
                    ref_cam, src_cams = proj_matrices[:, -1, ...], proj_matrices[:, :-1, ...]
                    reproj_xyd, in_range = depth_fusion.get_reproj(ref_depth, src_depths, ref_cam, src_cams)
                    vis_masks, vis_mask = depth_fusion.vis_filter(ref_depth, reproj_xyd, in_range,
                                                                droid_visualization.thresh_disp,
                                                                droid_visualization.filter_thresh, droid_visualization.thresh_view)
                    ref_depth_avg = depth_fusion.ave_fusion(ref_depth, reproj_xyd, vis_masks)
                    ref_image = video.ref_image[dirty_index[-1]].permute(1, 2, 0)
                    idx_img = depth_fusion.get_pixel_grids(*ref_depth_avg.size()[-2:]).unsqueeze(0)
                    idx_cam = depth_fusion.idx_img2cam(idx_img, ref_depth_avg, ref_cam)
                    points = depth_fusion.idx_cam2world(idx_cam, ref_cam)[..., :3, 0]
                    ref_pose = torch.inverse(poses[-1]).cpu().numpy()
                    ix = dirty_index[-1].item()
                    if len(droid_visualization.cameras.keys()) > 1:
                        if ix in droid_visualization.cameras:
                            vis.remove_geometry(droid_visualization.cameras[ix])
                            del droid_visualization.cameras[ix]

                        if ix in droid_visualization.points:
                            vis.remove_geometry(droid_visualization.points[ix])
                            del droid_visualization.points[ix]

                    ### add camera actor ###
                    cam_actor = create_camera_actor(True)
                    cam_actor.transform(ref_pose)
                    
                    vis.add_geometry(cam_actor, reset_bounding_box=False)
                    droid_visualization.cameras[ix] = cam_actor
                    droid_visualization.poses[ix] = ref_pose
                    
                    mask = vis_mask.reshape(-1)
                    pts = points.reshape(-1, 3)[mask].cpu().numpy()
                    clr = ref_image.reshape(-1, 3)[mask].cpu().numpy()

                    ## add point actor ###
                    point_actor = create_point_actor(pts, clr)
                    # if len(pts):
                    #     point_actor = down_sample(point_actor)
                    vis.add_geometry(point_actor, reset_bounding_box=False)
                    droid_visualization.points[ix] = point_actor
                    
                    if len(droid_visualization.cameras) >= droid_visualization.warmup:
                        cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

                    droid_visualization.ix += 1
                    vis.poll_events()
                    vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()