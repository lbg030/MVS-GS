from droid_slam.v_fuse_noise_filter.src.config import load_config
from droid_slam.v_fuse_noise_filter.src.vfuse import VFUSE
from droid_slam.v_fuse_noise_filter.src.tools.consensus_filtering import consensus_filtering

from droid_slam.visualization_utils import *

class NoiseFilter():
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.v_fuse_config)
        self.config['mode'] = "inference"
        self.vfuse = VFUSE(self.config)
    
    def inference(self, video,  proj_matrices, v_fuse_img_ids, images, depth_values):
        depthmap = video.disps_up[v_fuse_img_ids]
        conf_maps = video.confs_up[v_fuse_img_ids]
        K = proj_matrices[0][0][1]
        ref_index = int(v_fuse_img_ids[0])
        
        # near_depth = video.disps_up.min()
        # far_depth = video.disps_up.max()
        
        # near_depth = depth_values.min()
        near_depth = torch.tensor([0], device=depth_values.device)
        far_depth = depth_values.max()
        
        outputs = self.vfuse.inference(depthmap=depthmap, images=images, proj_matrices=proj_matrices, confs=conf_maps, K=K, ref_index=ref_index, near_depth=near_depth, far_depth=far_depth)
        depth_est_averaged, v_fuse_conf_map = consensus_filtering(self.config, outputs, depthmap, images, proj_matrices)
        # depth_est_avraged = outputs['fused_depth']
        # confs = outputs['fused_conf']
        # mask = confs > 0.5
        # depth_est_averaged = depth_est_avraged * mask

        return depth_est_averaged