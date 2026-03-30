import torch
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from trajectory_filler import PoseTrajectoryFiller
from droid_backend import DroidBackend
from collections import OrderedDict
from torch.multiprocessing import Process, Queue

from mvs_utils import model_load
from utils.disps_utils import last_calculate_disps_up

from init_gaussian_train import  GaussianTrain, gauss_train_run
from arguments import PipelineParams
from argparse import ArgumentParser
from gui import gui_utils, slam_gui
import time


class Droid:
    def __init__(self, args, config, tensor_img, original_image):
        super(Droid, self).__init__()
        self.config = config
        self.load_weights(config['model_params']['weight'])
        self.mvsnet = model_load(args) if config['model_params']['resume'] is not None else None
        if self.mvsnet:
            self.mvsnet.eval()
        self.use_gui = self.config["Visualize"]['use_gui']
        
        self.image_size = config['tracking']['image_size']
        self.depth_fusion_size = config['tracking']['depth_fusion_size']
        self.buffer = config['tracking']['buffer']
        self.stereo = args.stereo

        self.motion_filter_thresh = config['tracking']['motion_filter_thresh']
        self.device = config['model_params']['data_device']
        self.white_background = config['model_params']['white_background']
        
        self.q_main2vis = Queue()
        self.q_vis2main = Queue()
        self.q_front2gs = Queue()
        self.q_gs2main = Queue()
        
        self.video = DepthVideo(config, self.image_size, self.buffer, stereo=self.stereo, depth_size=self.depth_fusion_size)
        self.video.downed_img = torch.stack(tensor_img)

        self.filterx = MotionFilter(self.net, self.video, thresh=self.motion_filter_thresh)
        self.frontend = DroidFrontend(self.net, self.video, self.mvsnet, args, self.config, self.q_front2gs, self.q_gs2main)

        if self.use_gui:
            self.gaussians = GaussianTrain(self.q_front2gs, self.q_main2vis, self.q_gs2main, self.video, self.config)
            self.visualization()

        self.front2gs_multi_process()

        if self.use_gui:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.config, self.video, self.q_main2vis, self.q_vis2main))
            self.visualizer.start()
        
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)
        self.backend = DroidBackend(self.net, self.video, self.mvsnet, self.config)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.start.record()
        
    def visualization(self):
        parser = ArgumentParser(description="Training script parameters")
        bg_color = torch.tensor([0,0,0], device=self.device) if not self.white_background else torch.tensor([1,1,1], device=self.device)
        bg_color = bg_color.to(torch.float32)
        
        params_gui = gui_utils.ParamsGUI(
            pipe=PipelineParams(parser, self.config),
            background=bg_color,
            gaussians=self.gaussians.gaussians,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )
    
        self.gui_process = Process(target=slam_gui.run, args=(params_gui,))
        self.gui_process.start()

    def front2gs_multi_process(self):
        self.front2gs_process = Process(target=gauss_train_run, args=(self.q_front2gs, self.q_main2vis, self.q_gs2main, self.video, self.config))
        self.front2gs_process.start()
        time.sleep(5)

    def load_weights(self, weights):
        """ load trained model weights """
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)
            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        ref_id = self.frontend.t1 - 3
        last_calculate_disps_up(self.frontend, ref_id)
        
        del self.frontend
        del self.filterx
        
        sampled_tstamps = [i for i in range(self.video.downed_img.shape[0])]
        sampled_images = self.video.downed_img[sampled_tstamps]
        sampled_intrinsic = self.video.intrinsics[ref_id].repeat(len(sampled_tstamps), 1) * 8
        
        camera_trajectory = self.traj_filler([sampled_tstamps, sampled_images, sampled_intrinsic])
        sampled_trajectory = camera_trajectory.data
        
        self.q_front2gs.put(["break", self.config['mapping']['final_iters'], [sampled_images, sampled_trajectory, sampled_tstamps]])
        
        while True:
            if not self.q_gs2main.empty():
                msg = self.q_gs2main.get()
                if msg == "final":
                    torch.cuda.synchronize()  # Ensure all CUDA operations are completed
                    self.end.record()
                    time.sleep(1)
                    elapsed_time = self.start.elapsed_time(self.end)
                    FPS = self.config['Dataset']['total_img_len'] / (elapsed_time * 0.001) if elapsed_time > 0 else float('inf')
                    print("--------------------------------")
                    print(f"FPS : {FPS}")
                    print("--------------------------------")
                    
                if msg == "done": 
                    break
        
        with open("results.txt", "a") as f:
            f.write(f"FPS : {FPS}\n")
            f.write("\n")
        
        sampled_tstamps = [i for i in range(self.video.downed_img.shape[0])]
        sampled_images = self.video.downed_img[sampled_tstamps]
        sampled_intrinsic = self.video.intrinsics[0].repeat(len(sampled_tstamps), 1) * 8
        
        camera_traj = self.traj_filler([sampled_tstamps, sampled_images, sampled_intrinsic])
        sampled_trajectory = camera_traj.data
        
        # np.save("trajectory.npy", sampled_trajectory)
        
        return 0
