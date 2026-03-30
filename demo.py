import sys
sys.path.append('droid_slam')
from droid_slam.droid import Droid
from tqdm import tqdm
import yaml
from droid_slam.utils.config_utils import load_config
import torch
import argparse
from droid_slam.utils.droid_utils import show_image, image_stream

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    """ 명령줄 인수를 파싱합니다. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--mvs_weight', default="droid_slam/config/config_mvsformer.json", type=str, help='config file path (default: None)')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument("--resume", default="None")
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--ndepths', type=str, default=None, help='ndepths')
    parser.add_argument('--depth_interals_ratio', type=str, default=None, help='depth_interals_ratio')

    return parser.parse_args()

def configure_debug_mode(args):
    args.resume = "weights/mvsformer.pth"
    return args

def main():
    args = parse_arguments()
    args.stereo = False
    args.stride = 1
    # if __debug__: 
        # args.config = "config/TUM/freiburg3_long_office_household.yaml"
        # args.config = "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/scannet/scene0000.yaml"
        # args = configure_debug_mode(args)

    # args.config = "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/T&T/barn.yaml"
    args.config = "config/replica/office0.yaml"
    # args.config = "config/aerial/scenario3.yaml"
    # args.config = "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/TUM/freiburg3_long_office_household.yaml"
    
    
    args = configure_debug_mode(args)
    
    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)
    
    config = load_config(args.config)
    print(f"config: {args.config}")
    if args.eval:
        config['Visualize']['eval_rendering'] = True
        config['Visualize']['use_gui'] = True
        
    torch.multiprocessing.set_start_method('spawn')

    droid = None
    tstamps = []
    config, stream_img, intrinsics, original_image = image_stream(config)

    time_step = stream_img[0]
    tensor_img = stream_img[1]
    
    for (t, image) in tqdm(zip(time_step, tensor_img)):
        if t < config['model_params']['t0']:
            continue
        
        if not config['Visualize']['disable_vis']:
            show_image(image[0])
            
        if droid is None:
            droid = Droid(args, config, tensor_img, original_image)
            
        droid.track(t, image, intrinsics=intrinsics)
    droid.terminate([time_step, tensor_img, intrinsics])

if __name__ == '__main__':
    main()