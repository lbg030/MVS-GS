import torch

from factor_graph import FactorGraph
from trajectory_filler import PoseTrajectoryFiller
from utils.disps_utils import calculate_disps_up
from mvs_utils import mvs_warmed_up

class DroidFrontend:
    def __init__(self, net, video, mvsnet, args, config, q_front2gs, q_gs2main = None):
        
        self.config = config
        self.video = video
        self.device = self.config['model_params']['data_device']
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48)
        self.pose_filler = PoseTrajectoryFiller(net, video)
        self.mvsnet = mvsnet
        self.tmp = self.config['model_params']['tmp']
        self.args = args
        
        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.is_warmped_up = False
        self.count = 0

        self.max_age = self.config['tracking']['max_age']
        self.iters1 = self.config['tracking']['iters1']
        self.iters2 = self.config['tracking']['iters2']

        self.warmup = self.config['tracking']['warm_up']
        self.beta = self.config['tracking']['beta']
        self.frontend_nms = self.config['tracking']['frontend_nms']
        self.keyframe_thresh = self.config['tracking']['keyframe_thresh']
        self.frontend_window = self.config['tracking']['frontend_window']
        self.frontend_thresh = self.config['tracking']['frontend_thresh']
        self.frontend_radius = self.config['tracking']['frontend_radius']
        self.max_length = self.config['tracking']['max_length']
        
        self.backend_thresh = self.config['tracking']['backend_thresh']
        self.backend_radius = self.config['tracking']['backend_radius']
        self.backend_nms = self.config['tracking']['backend_nms']
        
        # GS
        self.scene_init = True
        self.q_front2gs = q_front2gs
        self.q_gs2main = q_gs2main
        self.top_10_nvs_loss = []
        
        self.set_hyperparams()
    
    def set_hyperparams(self):
        fx = self.config['Dataset']['Calibration']['fx']
        fy = self.config['Dataset']['Calibration']['fy']
        cx = self.config['Dataset']['Calibration']['cx']
        cy = self.config['Dataset']['Calibration']['cy']

        h0 = self.config['Dataset']['Calibration']['height']
        w0 = self.config['Dataset']['Calibration']['width']
        h1 = self.config['tracking']['depth_fusion_size'][0]
        w1 = self.config['tracking']['depth_fusion_size'][1]
        
        fx, cx = fx * (w1 / w0), cx * (w1 / w0)
        fy, cy = fy * (h1 / h0), cy * (h1 / h0)
        
        self.intr_mat = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

    def __initialize(self):
        """ initialize the SLAM system """
        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __update(self):
        """ add edges, perform update """
        is_keyframe_remove = False
        
        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
                                            self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)
            
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            is_keyframe_remove = True
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # refine depths
        if not self.is_warmped_up:
            mvs_warmed_up(self)
            self.is_warmped_up = True
        
        i = 3
        if not is_keyframe_remove:
            ref_id = self.t1-i        

            calculate_disps_up(self, i)

            if self.scene_init:
                self.start_idx = ref_id
                self.q_front2gs.put(["init", [ref_id-3, ref_id+1], self.intr_mat, ref_id])
                self.scene_init = False
            else:
                self.q_front2gs.put(["update", [ref_id-3, ref_id+1], ref_id])
            
            if self.t1 % 30 == 0:
                self.update_lowmem()

                sampled_tstamps = [i for i in range(int(self.video.tstamp[ref_id].item()))] if ref_id < 30 else [i for i in range(int(self.video.tstamp[ref_id-30].item()), int(self.video.tstamp[ref_id].item()))]
                
                if self.top_10_nvs_loss:
                    sampled_tstamps += self.top_10_nvs_loss
                    
                sampled_images = self.video.downed_img[sampled_tstamps]
                sampled_intrinsic = self.video.intrinsics[ref_id].repeat(len(sampled_tstamps), 1) * 8
                
                camera_trajectory = self.pose_filler([sampled_tstamps, sampled_images, sampled_intrinsic])
                sampled_trajectory = camera_trajectory.data
                self.q_front2gs.put(["GBA", sampled_images, sampled_trajectory, sampled_tstamps])
                
                while True:
                    if not self.q_gs2main.empty():
                        msg = self.q_gs2main.get()
                        if msg[0] == "done":
                            self.top_10_nvs_loss = msg[1]
                            break
        
        i = 3
        calculate_disps_up(self, i)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        self.video.dirty[max(self.graph.ii.min(), self.t1-i-3):(self.t1-i+1)] = True

    @torch.no_grad()
    def update_lowmem(self, steps=12):
        """ main update """
        t = self.video.counter.value
        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t)
        print()
        print(f"update_lowmem: {steps}")
        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        print(f"update_lowmem: {steps} done")
        return True

    def __call__(self):
        """ main update """
        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            self.update_lowmem(steps=12)
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
