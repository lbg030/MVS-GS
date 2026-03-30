import numpy as np
import cv2
import os
import torch

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)
    
def image_stream(config):
    """ image generator """

    # imagedir,calib, stride = args.imagedir, args.calib, args.stride
    # calib = np.loadtxt(calib, delimiter=" ")
    # fx, fy, cx, cy = calib[:4]
    
    imagedir = config['Dataset']['dataset_path']
    stride = config['Dataset']['stride']
    
    fx= config['Dataset']['Calibration']['fx']
    fy= config['Dataset']['Calibration']['fy']
    cx= config['Dataset']['Calibration']['cx']
    cy= config['Dataset']['Calibration']['cy']
    
    k1 = config['Dataset']['Calibration']['k1']
    k2 = config['Dataset']['Calibration']['k2']
    p1 = config['Dataset']['Calibration']['p1']
    p2 = config['Dataset']['Calibration']['p2']
    k3 = config['Dataset']['Calibration']['k3']
    
    distortion_list = [k1, k2, p1, p2, k3]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    if config['Dataset']['scene_name'].startswith("scannet"):
        image_list = sorted(os.listdir(imagedir), key=lambda x: int(os.path.splitext(x)[0]))[::stride]
    else :
        image_list = sorted(os.listdir(imagedir))[::stride]
    config['Dataset']['total_img_len'] = len(image_list)
    
    # h1, w1 = args.image_size[0], args.image_size[1] 
    # h1 = 256 #int(h0 * np.sqrt((256 * 448) / (h0 * w0)))
    # w1 = 448 #int(w0 * np.sqrt((256 * 448) / (h0 * w0)))
    
    h1 = config['tracking']['image_size'][0]
    w1 = config['tracking']['image_size'][1]
    
    t_list = []
    tensor_img = []
    original_img = []
    print(f"Image length: {len(image_list)}")
    
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        # if len(calib) > 4:
        #     image = cv2.undistort(image, K, calib[4:])
        if all(value != 0 for value in distortion_list):
            image = cv2.undistort(image, K, np.array(distortion_list))

        h0, w0, _ = image.shape
        if t == 0:
            # args.original_img_size = [h0, w0]
            config['Dataset']['Calibration']['width'] = w0
            config['Dataset']['Calibration']['height'] = h0
            
            # args.calib_data = [fx, fy, cx, cy]
            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)
        
        original_img.append(torch.as_tensor(image).permute(2, 0, 1)[None])
        image = cv2.resize(image, (w1, h1))
        # image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        t_list.append(t)
        tensor_img.append(image[None])

        # yield args, [t, image[None], intrinsics]
    return config, [t_list, tensor_img], intrinsics, original_img