import torch

from lietorch import SE3
from factor_graph import FactorGraph
from droid_slam.data_readers.mvsnet_dataloader import mvs_loader
from utils.image_utils import average_edge_pixels
from utils.mvs_depth_utils import run_mvs_depth_postprocess
    
class DroidBackend:
    def __init__(self, net, video, mvsnet, args):
        self.video = video
        self.update_op = net.update
        self.mvsnet = mvsnet
        self.args = args

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.beta = args["tracking"]["beta"]
        self.backend_thresh = args["tracking"]["backend_thresh"]
        self.backend_radius = args["tracking"]["backend_radius"]
        self.backend_nms = args["tracking"]["backend_nms"]

    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """
        t = self.video.counter.value
        if not self.video.stereo and not torch.any(self.video.disps_sens):
                self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t)

        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        
        if self.mvsnet is not None:                
            all_poses = SE3(self.video.poses).matrix()

            for t1 in range(2, t-2):
                ref_id, src_ids = t1, [t1 - 2, t1 - 1, t1 + 1, t1 + 2]
                img_ids = [ref_id] + src_ids
                poses = all_poses[img_ids]
                tstamps = self.video.tstamp[img_ids]
                images, proj_matrices, depth_values = mvs_loader(self.args, tstamps, poses, self.video.disps[ref_id])
                final_depth = run_mvs_depth_postprocess(
                    mvsnet=self.mvsnet,
                    images=images,
                    proj_matrices=proj_matrices,
                    depth_values=depth_values,
                    disparity_map=average_edge_pixels(self.video.disps[ref_id]),
                    tmp=self.args["model_params"]["tmp"],
                    max_length=3.0,
                    use_confidence_mask=True,
                )

                self.video.disps_up[ref_id] = final_depth.squeeze(0)
                self.video.dirty[max(0, ref_id - 3):(ref_id+1)] = True
