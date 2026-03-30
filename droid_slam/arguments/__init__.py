#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self.exp_set = "00"
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser, config):
        self.convert_SHs_python = config['pipeline_params']['convert_SHs_python']
        self.compute_cov3D_python = config['pipeline_params']['compute_cov3D_python']
        self.debug = config['pipeline_params']['debug']
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser, config):
        self.iterations = config['mapping']['opt_params']['iterations']
        self.position_lr_init = config['mapping']['opt_params']['position_lr_init']
        self.position_lr_final = config['mapping']['opt_params']['position_lr_final']
        self.position_lr_delay_mult = config['mapping']['opt_params']['position_lr_delay_mult']
        self.position_lr_max_steps = config['mapping']['opt_params']['position_lr_max_steps']
        self.feature_lr = config['mapping']['opt_params']['feature_lr']
        self.opacity_lr = config['mapping']['opt_params']['opacity_lr']
        self.scaling_lr = config['mapping']['opt_params']['scaling_lr']
        self.rotation_lr = config['mapping']['opt_params']['rotation_lr']
        self.percent_dense = config['mapping']['opt_params']['percent_dense']
        self.densification_interval = config['mapping']['opt_params']['densification_interval']
        self.opacity_reset_interval = config['mapping']['opt_params']['opacity_reset_interval']
        self.densify_from_iter = config['mapping']['opt_params']['densify_from_iter']
        self.densify_until_iter = config['mapping']['opt_params']['densify_until_iter']
        self.densify_grad_threshold = config['mapping']['opt_params']['densify_grad_threshold']
        
        # New
        self.shape_lr = config['mapping']['opt_params']['shape_lr']
        self.shape_reset_interval = config['mapping']['opt_params']['shape_reset_interval']
        self.shape_pruning_interval = config['mapping']['opt_params']['shape_pruning_interval']
        self.prune_shape_threshold = config['mapping']['opt_params']['prune_shape_threshold']
        self.prune_opacity_threshold = config['mapping']['opt_params']['prune_opacity_threshold']
        self.shape_strngth = config['mapping']['opt_params']['shape_strngth']
        self.im_laplace_scale_factor = config['mapping']['opt_params']['im_laplace_scale_factor']
        self.lambda_im_laplace = config['mapping']['opt_params']['lambda_im_laplace']

        self.lambda_dssim = config['mapping']['opt_params']['lambda_dssim']
        self.labmda_l1 = config['mapping']['opt_params']['labmda_l1']
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
