import torch
import torch_scatter
import pytorch3d.transforms as p3dt
from .transform import (
    transform_pc,
    quaternion_to_euler,
)
from pytorch3d.loss.chamfer import chamfer_distance


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    nan_mask = torch.isnan(loss_per_part)
    loss_per_part[nan_mask] = 0.0
    valids = valids.float().detach()
    loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    return loss_per_data


def trans_metrics(trans1, trans2, valids, metric):
    """Evaluation metrics for translation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3], pred translation
        trans2: [B, P, 3], gt translation
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ["mse", "rmse", "mae"]
    if metric == "mse":
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == "rmse":
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1) ** 0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def rot_metrics(rot1, rot2, valids, metric):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        rot1: [B, P, 4], pred quat
        rot2: [B, P, 4], gt quat
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ["mse", "rmse", "mae"]
    deg1 = quaternion_to_euler(rot1, to_degree=True)  # [B, P, 3]
    deg2 = quaternion_to_euler(rot2, to_degree=True)

    diff1 = (deg1 - deg2).abs()
    diff2 = 360.0 - (deg1 - deg2).abs()
    # since euler angle has the discontinuity at 180
    diff = torch.minimum(diff1, diff2)
    if metric == "mse":
        metric_per_data = diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == "rmse":
        metric_per_data = diff.pow(2).mean(dim=-1) ** 0.5
    else:
        metric_per_data = diff.abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids):
    """Compute the `Part Accuracy` in the paper.

    We compute the per-part chamfer distance, and the distance lower than a
        threshold will be considered as correct.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3], pred_translation
        trans2: [B, P, 3], gt_translation
        rot1: [B, P, 4], Rotation3D, quat or rmat
        rot2: [B, P, 4], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], accuracy per data in the batch
    """
    B, P = pts.shape[:2]

    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)

    pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
    pts2 = pts2.flatten(0, 1)
    loss_per_data, _ = chamfer_distance(
        x=pts1,
        y=pts2,
        single_directional=False,
        point_reduction="mean",
        batch_reduction=None,
    )  # [B*P, N]
    loss_per_data = loss_per_data.view(B, P).type_as(pts)

    # part with CD < `thre` is considered correct
    thre = 0.01
    acc_per_part = (loss_per_data < thre) & (valids == 1)
    # the official code is doing avg per-shape acc (not per-part)
    acc = acc_per_part.sum(-1) / (valids == 1).sum(-1)
    return acc, acc_per_part, loss_per_data


@torch.no_grad()
def calc_part_acc_weighted(
    pts: torch.Tensor,  # [B, N_sum, 3]
    gt_trans: torch.Tensor,  # [valid_P, 3]
    gt_rots: torch.Tensor,  # [valid_P, 4]
    pred_trans: torch.Tensor,  # [valid_P, 3]
    pred_rots: torch.Tensor,  # [valid_P, 4]
    points_per_part: torch.Tensor,  # [B, P]
    part_valids: torch.Tensor,  # [B, P]
    part_valids_wo_redundancy: torch.Tensor,  # [B, P]
):
    B, P = part_valids.shape
    points_per_valid_part = points_per_part[part_valids]
    gt_trans_point = gt_trans.repeat_interleave(
        points_per_valid_part, dim=0
    )  # (B*N_sum, 3)
    gt_rots_point = gt_rots.repeat_interleave(points_per_valid_part, dim=0)
    pred_trans_point = pred_trans.repeat_interleave(points_per_valid_part, dim=0)
    pred_rots_point = pred_rots.repeat_interleave(points_per_valid_part, dim=0)
    pts_gt = (
        p3dt.quaternion_apply(gt_rots_point, pts.view(-1, 3)) + gt_trans_point
    ).detach()  # (B*N_sum, 3)
    pts_pred = (
        p3dt.quaternion_apply(pred_rots_point, pts.view(-1, 3)) + pred_trans_point
    ).detach()  # (B*N_sum, 3)

    # padding to (valid_P, N_max, 3)
    N_max = points_per_valid_part.max()
    valid_P = points_per_valid_part.shape[0]
    pts_gt_padded = torch.zeros(valid_P, N_max, 3, device=pts.device)
    pts_pred_padded = torch.zeros(valid_P, N_max, 3, device=pts.device)

    # Create row indices
    row_idx = torch.arange(valid_P, device=pts.device).unsqueeze(1).expand(-1, N_max)
    # Create column indices
    col_idx = torch.arange(N_max, device=pts.device).unsqueeze(0).expand(valid_P, -1)
    mask = col_idx < points_per_valid_part.unsqueeze(1)
    source_idx = torch.arange(pts_gt.shape[0], device=pts.device)
    pts_gt_padded[row_idx[mask], col_idx[mask]] = pts_gt[source_idx]
    pts_pred_padded[row_idx[mask], col_idx[mask]] = pts_pred[source_idx]

    # Compute chamfer distance
    shape_cd, _ = chamfer_distance(
        x=pts_gt_padded,
        y=pts_pred_padded,
        x_lengths=points_per_valid_part,
        y_lengths=points_per_valid_part,
        single_directional=False,
        point_reduction="mean",
        batch_reduction=None,
    )  # (valid_P,)

    # Compute part accuracy
    threshold = 0.01
    acc_per_part = (shape_cd < threshold).float()

    # Following way of calculation is used before we added redundancy part
    # When redundancy part is not added, two methods should be equivalent
    # part_offset = torch.cat(
    #     [torch.tensor([0], device=part_valids.device), part_valids.sum(-1).cumsum(0)]
    # )
    # acc_per_data = torch_scatter.segment_csr(
    #     src=acc_per_part,
    #     indptr=part_offset,
    #     reduce="mean",
    # )

    # Recover to object (B, P, P)
    acc_per_part_padded = torch.zeros(B, P, device=part_valids.device)
    acc_per_part_padded[part_valids] = acc_per_part
    acc_per_part_padded[~part_valids_wo_redundancy] = 0.0
    acc_per_data = acc_per_part_padded.sum(-1) / part_valids_wo_redundancy.sum(-1)

    return acc_per_data


@torch.no_grad()
def calc_shape_cd(pts, trans1, trans2, rot1, rot2, valids):

    B, P, N, _ = pts.shape

    valid_mask = valids[..., None, None]  # [B, P, 1, 1]

    pts = pts.detach().clone()

    pts = pts.masked_fill(valid_mask == 0, 1e3)

    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)

    shape1 = pts1.flatten(1, 2)
    shape2 = pts2.flatten(1, 2)

    shape_cd, _ = chamfer_distance(
        x=shape1,
        y=shape2,
        single_directional=False,
        point_reduction=None,
        batch_reduction=None,
    )
    shape_cd = shape_cd[0] + shape_cd[1]

    shape_cd = shape_cd.view(B, P, N).mean(-1)
    shape_cd = _valid_mean(shape_cd, valids)

    return shape_cd


def calc_shape_cd_weighted(
    pts: torch.Tensor,  # [B, N_sum, 3]
    gt_trans: torch.Tensor,  # [valid_P, 3]
    gt_rots: torch.Tensor,  # [valid_P, 4]
    pred_trans: torch.Tensor,  # [valid_P, 3]
    pred_rots: torch.Tensor,  # [valid_P, 4]
    points_per_part: torch.Tensor,  # [B, P]
    part_valids: torch.Tensor,  # [B, P]
    part_valids_wo_redundancy: torch.Tensor,  # [B, P]
):
    B, N_sum, _ = pts.shape
    points_per_valid_part = points_per_part[part_valids]
    gt_trans_point = gt_trans.repeat_interleave(
        points_per_valid_part, dim=0
    )  # (B*N_sum, 3)
    gt_rots_point = gt_rots.repeat_interleave(points_per_valid_part, dim=0)
    pred_trans_point = pred_trans.repeat_interleave(points_per_valid_part, dim=0)
    pred_rots_point = pred_rots.repeat_interleave(points_per_valid_part, dim=0)

    pts_gt = (
        p3dt.quaternion_apply(gt_rots_point, pts.view(-1, 3)) + gt_trans_point
    ).detach()  # (B*N_sum, 3)
    pts_pred = (
        p3dt.quaternion_apply(pred_rots_point, pts.view(-1, 3)) + pred_trans_point
    ).detach()  # (B*N_sum, 3)

    # Handle redundancy parts
    # The logic here is that we do not consider the chamfer distance for the
    # redundant parts, so we set the distance to be a large number
    # and the redundant parts will not contribute to the final distance
    # and we make sure for redundant parts, thier chamfer distance is 0
    if (part_valids_wo_redundancy != part_valids).any():
        redundancy_parts_mask = part_valids & ~part_valids_wo_redundancy  # (B, P)
        redundancy_parts_mask = redundancy_parts_mask[part_valids]  # (valid_P,)
        redundancy_parts_points_mask = redundancy_parts_mask.repeat_interleave(
            points_per_valid_part, dim=0
        )
        pts_gt[redundancy_parts_points_mask] = 1e3
        pts_pred[redundancy_parts_points_mask] = 1e3

    # Back to the original shape
    pts_gt = pts_gt.view(B, N_sum, -1)
    pts_pred = pts_pred.view(B, N_sum, -1)

    shape_cd, _ = chamfer_distance(
        x=pts_gt,
        y=pts_pred,
        single_directional=False,
        point_reduction=None,
        batch_reduction=None,
    )
    shape_cd = shape_cd[0] + shape_cd[1]  # (B, N_sum)
    # mean over parts
    offset = torch.cat(
        [
            torch.zeros((B, 1), device=points_per_part.device),
            points_per_part.cumsum(-1),  # (B, P)
        ],
        dim=-1,
    ).long()
    shape_cd = torch_scatter.segment_csr(
        src=shape_cd,
        indptr=offset,
        reduce="mean",
    )  # (B, P)

    shape_cd = _valid_mean(shape_cd, part_valids_wo_redundancy)

    return shape_cd
