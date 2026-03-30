import open3d as o3d
import numpy as np
import torch

def export_pointcloud(droid_visualization, file_path="test.ply"):
    print("Exporting point cloud to", file_path)
    merged_pc = o3d.geometry.PointCloud()
    
    for point in sorted(droid_visualization.points.keys()):
        pc2 = droid_visualization.points[point]
        
        merged_pc.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(merged_pc.points), np.asarray(pc2.points)))
        )
        
        if pc2.colors:
            merged_pc.colors = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(merged_pc.colors), np.asarray(pc2.colors)))
            )
        
        if pc2.normals:
            merged_pc.normals = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(merged_pc.normals), np.asarray(pc2.normals)))
            )
    return merged_pc
    
def save_pointcloud(pcd=None, points=None, colors=None, name="test.ply"):
    import open3d as o3d
    cloud_points = o3d.geometry.PointCloud()
    if points is not None:
        if isinstance(points, (torch.Tensor, torch.nn.Parameter)):
            points = points.squeeze() if len(points.shape) == 3 else points
            colors = colors.squeeze() if len(colors.shape) == 3 else colors
            
            points = points.detach().cpu().numpy()
            colors = colors.detach().cpu().numpy()
            
            cloud_points.points = o3d.utility.Vector3dVector(points)
            cloud_points.colors = o3d.utility.Vector3dVector(colors)
        
    elif pcd is not None :
        cloud_points.points = o3d.utility.Vector3dVector(pcd.points)
        cloud_points.colors = o3d.utility.Vector3dVector(pcd.colors)
    
    else :
        assert False, "No points to save"
        
    o3d.io.write_point_cloud(name, cloud_points)
    print(f"Point cloud saved to {name}")

def show_plot(frame, title=None, cmap="jet",figsize=[5,5]):

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    frame = frame.squeeze()
    if title is not None:
        plt.title(title)
    if isinstance(frame, torch.Tensor):
        plt.imshow(frame.detach().cpu().numpy(), cmap=cmap) if frame.shape[0] != 3 else plt.imshow(frame.permute(1,2,0).detach().cpu().numpy())
    
    elif isinstance(frame, np.ndarray):
        plt.imshow(frame, cmap=cmap) if frame.shape[0] != 3 else plt.imshow(frame.transpose(1,2,0)) 
        
    else :
        assert False, "Invalid frame type"
        
        

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    
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


    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor



def merge_point_cloud(points, colors):
    merged_pc = o3d.geometry.PointCloud()

    for point in sorted(points.keys()):
        pc2 = points[point]
        colors2 = colors[point]
        
        merged_pc.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(merged_pc.points), pc2 ))
        )
        
        merged_pc.colors = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(merged_pc.colors), colors2))
        )
    
    return merged_pc

def save_depth(depth, name="test"):
    from PIL import Image
    # Tensor를 NumPy 배열로 변환
    img_array = depth.squeeze().detach().cpu().numpy()
    
    # 값을 0~255 범위로 정규화 (이미지를 저장할 때 필요)
    img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    img_normalized = (img_normalized * 255).astype(np.uint8)

    # 컬러맵 적용 ("jet")
    from matplotlib import cm
    colormap = cm.get_cmap("jet")
    img_colored = colormap(img_normalized)[:, :, :3]  # RGBA에서 RGB만 사용
    img_colored = (img_colored * 255).astype(np.uint8)

    # PIL 이미지로 변환 후 저장
    img = Image.fromarray(img_colored)
    img.save(f"{name}.png")
    

def save_img(image, name="test"):
    # 텐서를 넘파이 배열로 변환
    from PIL import Image
    # 텐서를 넘파이 배열로 변환
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 2 or image.shape[0] == 1:
            img_array = image.squeeze().detach().cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
        elif image.shape[0] == 3:
            img_array = image.permute(1, 2, 0).detach().cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='RGB')
        else:
            img_array = image.squeeze().detach().cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
    else:
        image *= 255
        array = image.astype(np.uint8)  # 이미지 데이터 형식으로 변환
        # 첫 번째 채널을 선택하여 2D로 변환
        img = Image.fromarray(array[0])
    # 흰색 여백 제거를 위해 bbox 사용
    # bbox = img.getbbox()
    # if bbox:
    #     img = img.crop(bbox)

    # 이미지를 파일로 저장
    img.save(f"{name}.png")
    # print(f"Image with no white borders saved to {name}.png")

def vis_pcd(pcd):
    if isinstance(pcd, list):
        o3d.visualization.draw_geometries(pcd)
    else :
        return o3d.visualization.draw_geometries([pcd])
    
def vis_gs(pc, cam=None, save=False, mask=None):
    from utils.sh_utils import eval_sh
    pcd = o3d.geometry.PointCloud()
    
    if mask is not None:
        shs_view = pc.get_features[mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz)[mask]
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        point = o3d.utility.Vector3dVector(pc.get_xyz[mask].detach().cpu().numpy())
        color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
        
        pcd.points = point
        pcd.colors = color
    
    else:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        point = o3d.utility.Vector3dVector(pc.get_xyz.detach().cpu().numpy())
        color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
        
        pcd.points = point
        pcd.colors = color
    
    if cam is not None:
        total_list = [pcd]
        if isinstance(cam, list):
            for c in cam:
                cam_actor = create_camera_actor(True)
                ref_pose = np.eye(4)
                R, T = c.R, c.T
                ref_pose[:3, :3] = R
                ref_pose[:3, 3] = T
                
                ref_pose = np.linalg.inv(ref_pose)
                cam_actor.transform(ref_pose)
                # cam_actor.rotate(R)
                # cam_actor.translate(T)
                total_list.append(cam_actor)
        else :
            cam_actor = create_camera_actor(True)
            c = cam
            ref_pose = np.eye(4)
            R, T = c.R, c.T
            ref_pose[:3, :3] = R
            ref_pose[:3, 3] = T
            
            ref_pose = np.linalg.inv(ref_pose)
            cam_actor.transform(ref_pose)
            # cam_actor.rotate(R)
            # cam_actor.translate(T)
            total_list.append(cam_actor)
        return o3d.visualization.draw_geometries(total_list)
    
    if save:
        o3d.io.write_point_cloud("test.ply", pcd)
        
    o3d.visualization.draw_geometries([pcd])