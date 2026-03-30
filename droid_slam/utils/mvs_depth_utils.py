import torch
import torch.nn.functional as F


def _build_confidence_mask(mvs_outputs):
    final_depth = mvs_outputs["refined_depth"]
    mask = torch.ones_like(final_depth, dtype=torch.bool)

    for stage, thresh_conf in zip(
        ["stage1", "stage2", "stage3", "stage4"], [0.1, 0.2, 0.3, 0.4]
    ):
        conf_stage = F.interpolate(
            mvs_outputs[stage]["photometric_confidence"].unsqueeze(1),
            (mask.size(1), mask.size(2)),
        ).squeeze(1)
        mask = mask & (conf_stage > thresh_conf)

    return mask


@torch.no_grad()
def run_mvs_depth_postprocess(
    mvsnet,
    images,
    proj_matrices,
    depth_values,
    disparity_map,
    tmp,
    max_length,
    use_confidence_mask=False,
):
    mvs_outputs = mvsnet(
        images,
        proj_matrices,
        depth_values,
        disparity_map=disparity_map,
        tmp=tmp,
    )

    final_depth = mvs_outputs["refined_depth"]
    if use_confidence_mask:
        mask = _build_confidence_mask(mvs_outputs)
        final_depth[~mask] = 1e-6

    final_depth[final_depth > max_length] = 0
    final_depth[final_depth < 0] = 0
    return final_depth
