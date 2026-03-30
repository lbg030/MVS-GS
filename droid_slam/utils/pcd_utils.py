import torch
import depth_fusion
import open3d as o3d
import numpy as np
from lietorch import SE3
import init_gaussian_train as GaussianTrain
import torch.nn.functional as F

def vis_pcd(gaussian_train : GaussianTrain):
    o3d.visualization.draw_geometries([gaussian_train.point_cloud])

def save_pcd(gaussian_train : GaussianTrain):
    o3d.io.write_point_cloud(f"{gaussian_train.config['Dataset']['scene_name']}.ply", gaussian_train.point_cloud)

def create_point_actor_from_indices(gaussian_train : GaussianTrain, ix_list):
    pts, clr = generate_pcd(gaussian_train, ix_list, init=True)
    if pts.shape[0] > gaussian_train.point_num_thresh:
        point_actor = create_point_actor(gaussian_train, pts, clr)
        return point_actor
    return None

def create_point_actor(gaussian_train : GaussianTrain, points, colors):
    """Open3D를 사용해 포인트 클라우드를 생성하고, 노이즈를 제거합니다."""
    # Open3D Point Cloud 생성
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    if gaussian_train.voxel_down:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=gaussian_train.voxel_size)
        
    return point_cloud

def generate_pcd(gaussian_train : GaussianTrain, dirty_idx_list, l1_mask=None, rendered_depth=None, init=False):
    start_idx, end_idx = dirty_idx_list
    device = gaussian_train.video.poses.device

    # 인덱스 생성 및 포즈, 깊이 정보 선택
    dirty_index = torch.arange(start_idx, end_idx, device=device)
    poses = SE3(gaussian_train.video.poses[dirty_index]).matrix()  # [N, 4, 4]
    depths = gaussian_train.video.disps_up[dirty_index].unsqueeze(1).unsqueeze(0)  # [1, N, 1, H, W]

    # 참조 이미지 및 내부 파라미터 설정
    ref_image = gaussian_train.video.ref_image[end_idx-1].permute(1, 2, 0)  # [H, W, 3]
    intr_matrices = gaussian_train.intr_mat.unsqueeze(0).expand(poses.size(0), -1, -1)  # [N, 4, 4]

    # 프로젝션 행렬 생성
    proj_matrices = torch.stack((poses, intr_matrices), dim=1).unsqueeze(0)  # [1, N, 2, 4, 4]

    # 참조 깊이 및 소스 깊이 분리
    ref_depth = depths[:, -1, ...]  # [1, 1, H, W]
    src_depths = depths[:, :-1, ...]  # [1, N-1, 1, H, W]
    ref_cam = proj_matrices[:, -1, ...]  # [1, 2, 4, 4]
    src_cams = proj_matrices[:, :-1, ...]  # [1, N-1, 2, 4, 4]

    # 재투영 좌표 및 가시성 마스크 계산
    reproj_xyd, in_range = depth_fusion.get_reproj(ref_depth, src_depths, ref_cam, src_cams)
    vis_masks, vis_mask = depth_fusion.vis_filter(
        ref_depth, reproj_xyd, in_range,
        gaussian_train.thresh_disp, gaussian_train.filter_thresh, gaussian_train.thresh_view
    )
    # 평균 깊이 계산
    ref_depth_avg = depth_fusion.ave_fusion(ref_depth, reproj_xyd, vis_masks)

    # 픽셀 그리드 생성 및 월드 좌표 변환
    H, W = ref_depth_avg.shape[-2:]
    idx_img = depth_fusion.get_pixel_grids(H, W).unsqueeze(0)
    idx_cam = depth_fusion.idx_img2cam(idx_img, ref_depth_avg, ref_cam)
    points_world = depth_fusion.idx_cam2world(idx_cam, ref_cam)[..., :3, 0]  # [1, H, W, 3]

    # 텐서 평탄화
    points_flat = points_world.view(-1, 3)
    colors_flat = ref_image.view(-1, 3)
    mask_flat = vis_mask.view(-1)

    ref_image_mask = (ref_image.sum(dim=-1) > 0).view(-1)  # RGB 값이 모두 0이면 False
    # l1_mask 처리
    if l1_mask is not None:
        l1_mask_resized = F.interpolate(
            l1_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='nearest'
        ).squeeze().bool()
        l1_mask_flat = l1_mask_resized.view(-1)
        mask_flat = mask_flat & l1_mask_flat

    mask_flat = mask_flat & ref_image_mask
    # 마스크 적용 및 데이터 CPU로 이동
    valid_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze()
    if valid_indices.numel() == 0:
        return np.array([]), np.array([])
    pts = points_flat[valid_indices].detach().cpu().numpy()
    clr = colors_flat[valid_indices].detach().cpu().numpy()

    return pts, clr