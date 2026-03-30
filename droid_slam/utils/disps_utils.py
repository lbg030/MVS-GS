from typing import TYPE_CHECKING

from utils.image_utils import average_edge_pixels
import torch
from lietorch import SE3
from droid_slam.data_readers.mvsnet_dataloader import mvs_loader
from utils.mvs_depth_utils import run_mvs_depth_postprocess

if TYPE_CHECKING:
    import droid_frontend as DroidFrontend

@torch.no_grad()
def calculate_disps_up(frontend: "DroidFrontend", i):
    ref_id, src_ids = frontend.t1 - i, [frontend.t1-i-2, frontend.t1-i-1, frontend.t1-i+1, frontend.t1-i+2]
    img_ids = [ref_id] + src_ids
    poses = SE3(frontend.video.poses[img_ids]).matrix()
    tstamps = frontend.video.tstamp[img_ids]
    images, proj_matrices, depth_values = mvs_loader(frontend.config, tstamps, poses, frontend.video.disps[ref_id])

    final_depth = run_mvs_depth_postprocess(
        mvsnet=frontend.mvsnet,
        images=images,
        proj_matrices=proj_matrices,
        depth_values=depth_values,
        disparity_map=average_edge_pixels(frontend.video.disps[ref_id]),
        tmp=frontend.tmp,
        max_length=frontend.max_length,
        use_confidence_mask=True,
    )
    
    frontend.video.ref_image[ref_id] = frontend.video.downed_img[int(tstamps[0].item())].squeeze()[[2, 1, 0], :, :] / 255.0
    frontend.video.disps_up[ref_id] = final_depth

@torch.no_grad()
def last_calculate_disps_up(frontend: "DroidFrontend", last_idx):
    for i in range(1,3):
        ref_id = last_idx + i
        
        if i == 1:
            src_ids = [ref_id-2, ref_id-1, ref_id+1]
        else :
            src_ids = [ref_id-2, ref_id-1]
            
        img_ids = [ref_id] + src_ids
        poses = SE3(frontend.video.poses[img_ids]).matrix()
        tstamps = frontend.video.tstamp[img_ids]
        images, proj_matrices, depth_values = mvs_loader(frontend.config, tstamps, poses, frontend.video.disps[ref_id])
        final_depth = run_mvs_depth_postprocess(
            mvsnet=frontend.mvsnet,
            images=images,
            proj_matrices=proj_matrices,
            depth_values=depth_values,
            disparity_map=average_edge_pixels(frontend.video.disps[ref_id]),
            tmp=frontend.tmp,
            max_length=frontend.max_length,
            use_confidence_mask=False,
        )
        
        frontend.video.disps_up[ref_id] = final_depth.squeeze(0)
        frontend.video.ref_image[ref_id] = frontend.video.downed_img[int(tstamps[0].item())].squeeze()[[2, 1, 0], :, :] / 255.0
        
        frontend.q_front2gs.put(["update", [ref_id-3, ref_id+1], ref_id, True])
