from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from pinn_hoi.common.rot import gather_points, rotation_error_rad


def _safe_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean() if x.numel() > 0 else torch.zeros((), device=x.device, dtype=x.dtype)
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * target + (1.0 - p) * (1.0 - target)
    at = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (at * (1.0 - pt).pow(gamma) * bce).mean()


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (prob * target).sum(dim=(-1, -2))
    union = prob.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def contact_distance_loss(batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], contact_thresh_m: float) -> torch.Tensor:
    dist = batch['endpoint_obj_min_dist'][:, :-1]  # [B,T-1,2,21]
    prob = out['contact_prob']
    # Predicted contact should be close to surface; non-contact endpoints should not be forced to touch.
    close = prob * dist
    far_margin = (1.0 - prob) * F.relu(contact_thresh_m - dist)
    return close.mean() + 0.25 * far_margin.mean()


def no_contact_static_loss(batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> torch.Tensor:
    gt_contact = batch['contact_label'][:, :-1]
    contact_amount = gt_contact.flatten(2).mean(dim=-1)  # [B,T-1]
    no_contact = (contact_amount < 0.02).float()
    pred_delta_trans = torch.linalg.norm(out['delta_pose7'][..., 4:7], dim=-1)
    pred_delta_rot = torch.linalg.norm(out['delta_pose7'][..., 1:4], dim=-1)
    pred_delta_art = torch.abs(out['delta_pose7'][..., 0])
    return _safe_mean(pred_delta_trans + 0.2 * pred_delta_rot + 0.05 * pred_delta_art, no_contact)


def contact_velocity_loss(batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    hand = batch['hand_endpoints']             # [B,T,2,21,3]
    obj = batch['obj_points_world']            # [B,T,N,3]
    idx = batch['endpoint_nearest_obj_idx']     # [B,T,2,21]
    contact = batch['contact_label'][:, :-1]

    hand_v = hand[:, 1:] - hand[:, :-1]
    near_cur = gather_points(obj[:, :-1], idx[:, :-1])
    near_next = gather_points(obj[:, 1:], idx[:, :-1])
    obj_v = near_next - near_cur

    # Normal direction from object surface approximation to endpoint.
    normal = F.normalize(hand[:, :-1] - near_cur, dim=-1, eps=1e-6)
    rel = hand_v - obj_v
    normal_rel = torch.abs((rel * normal).sum(dim=-1))
    vel_loss = _safe_mean(normal_rel, contact)

    # Directional hand-driven motion metric/loss: object contact velocity should align with hand velocity.
    cos = F.cosine_similarity(hand_v, obj_v, dim=-1, eps=1e-6)
    align_loss = _safe_mean(1.0 - cos, contact)
    return vel_loss, align_loss


def impulse_dynamics_losses(batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obj_pose = batch['obj_pose7']
    hand = batch['hand_endpoints']
    obj = batch['obj_points_world']
    idx = batch['endpoint_nearest_obj_idx']
    contact_prob = out['contact_prob'].detach()  # make dynamics stable; contact head trained by BCE/focal separately
    impulse = out['impulse']

    gt_delta_trans = obj_pose[:, 1:, 4:7] - obj_pose[:, :-1, 4:7]
    gt_delta_rot = obj_pose[:, 1:, 1:4] - obj_pose[:, :-1, 1:4]

    weighted_impulse = contact_prob.unsqueeze(-1) * impulse
    total_impulse = weighted_impulse.sum(dim=(2, 3)) / 42.0
    # Let the network learn normalized impulse. Match direction and scale weakly to object displacement.
    dyn_trans = F.smooth_l1_loss(total_impulse, gt_delta_trans / max(dt, 1e-6), beta=0.05)

    near_cur = gather_points(obj[:, :-1], idx[:, :-1])
    center = obj_pose[:, :-1, 4:7].unsqueeze(2).unsqueeze(2)
    r = near_cur - center
    torque = torch.cross(r, weighted_impulse, dim=-1).sum(dim=(2, 3)) / 42.0
    dyn_ang = F.smooth_l1_loss(torque, gt_delta_rot / max(dt, 1e-6), beta=0.05)

    # Friction cone. Normal is approximated by object point -> endpoint direction.
    normal = F.normalize(hand[:, :-1] - near_cur, dim=-1, eps=1e-6)
    fn = (impulse * normal).sum(dim=-1)
    ft = impulse - fn.unsqueeze(-1) * normal
    mu = out['mu'].unsqueeze(-1).unsqueeze(-1)
    friction = F.relu(-fn).mean() + F.relu(torch.linalg.norm(ft, dim=-1) - mu * F.relu(fn)).mean()

    # Non-contact endpoints should not emit large force.
    non_contact = 1.0 - batch['contact_label'][:, :-1]
    non_contact_force = _safe_mean(torch.linalg.norm(impulse, dim=-1), non_contact)
    return dyn_trans, dyn_ang, friction, non_contact_force


def smoothness_losses(out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = out['delta_pose7']
    impulse = out['impulse']
    if delta.shape[1] <= 1:
        z = torch.zeros((), device=delta.device, dtype=delta.dtype)
        return z, z
    delta_smooth = (delta[:, 1:] - delta[:, :-1]).pow(2).mean()
    force_smooth = (impulse[:, 1:] - impulse[:, :-1]).pow(2).mean()
    return delta_smooth, force_smooth


def compute_losses(
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    cfg: Dict,
    physics_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    weights = cfg.get('loss_weights', {})
    dt = float(cfg.get('dt', 1.0 / 30.0))
    thresh = float(cfg.get('contact_thresh_m', 0.015))

    gt_next = batch['obj_pose7'][:, 1:]
    pred_next = out['pred_next_pose7']
    losses: Dict[str, torch.Tensor] = {}
    losses['obj_trans'] = F.smooth_l1_loss(pred_next[..., 4:7], gt_next[..., 4:7], beta=0.02)
    losses['obj_rot'] = rotation_error_rad(pred_next[..., 1:4], gt_next[..., 1:4]).mean()
    losses['obj_arti'] = F.smooth_l1_loss(pred_next[..., 0:1], gt_next[..., 0:1], beta=0.02)

    contact_gt = batch['contact_label'][:, :-1]
    losses['contact_bce'] = F.binary_cross_entropy_with_logits(out['contact_logits'], contact_gt)
    losses['contact_focal'] = focal_bce_with_logits(out['contact_logits'], contact_gt)
    losses['contact_dice'] = dice_loss(out['contact_prob'], contact_gt)
    losses['endpoint_contact_dist'] = contact_distance_loss(batch, out, thresh)

    losses['no_contact_static'] = no_contact_static_loss(batch, out)
    losses['contact_velocity'], losses['contact_motion_align'] = contact_velocity_loss(batch, out)
    losses['impulse_dyn'], losses['angular_dyn'], losses['friction_cone'], losses['non_contact_force'] = impulse_dynamics_losses(batch, out, dt)
    losses['delta_smooth'], losses['force_smooth'] = smoothness_losses(out)

    physics_keys = {
        'no_contact_static', 'contact_velocity', 'contact_motion_align', 'impulse_dyn',
        'angular_dyn', 'friction_cone', 'non_contact_force', 'force_smooth', 'delta_smooth',
        'endpoint_contact_dist',
    }
    total = torch.zeros((), device=pred_next.device, dtype=pred_next.dtype)
    for k, v in losses.items():
        w = float(weights.get(k, 0.0))
        if k in physics_keys:
            w *= float(physics_scale)
        total = total + w * v
    losses['total'] = total
    losses['physics_scale'] = torch.tensor(float(physics_scale), device=pred_next.device)
    return total, losses


def compute_metrics(batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    gt_next = batch['obj_pose7'][:, 1:]
    pred_next = out['pred_next_pose7']
    trans_err_m = torch.linalg.norm(pred_next[..., 4:7] - gt_next[..., 4:7], dim=-1)
    rot_err = rotation_error_rad(pred_next[..., 1:4], gt_next[..., 1:4])
    arti_err = torch.abs(pred_next[..., 0] - gt_next[..., 0])

    pred_contact = (out['contact_prob'] > 0.5).float()
    gt_contact = batch['contact_label'][:, :-1]
    tp = (pred_contact * gt_contact).sum()
    fp = (pred_contact * (1.0 - gt_contact)).sum()
    fn = ((1.0 - pred_contact) * gt_contact).sum()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)

    no_contact = (gt_contact.flatten(2).mean(dim=-1) < 0.02)
    pred_drift = torch.linalg.norm(out['delta_pose7'][..., 4:7], dim=-1)
    no_contact_drift = pred_drift[no_contact].mean() if no_contact.any() else torch.zeros((), device=pred_drift.device)

    vel_loss, align_loss = contact_velocity_loss(batch, out)
    contact_motion_cos = 1.0 - align_loss
    return {
        'obj_trans_err_m': trans_err_m.mean(),
        'obj_rot_err_rad': rot_err.mean(),
        'obj_arti_err': arti_err.mean(),
        'contact_precision': precision,
        'contact_recall': recall,
        'contact_f1': f1,
        'no_contact_drift_m': no_contact_drift,
        'contact_velocity_residual_m': vel_loss,
        'contact_motion_cos': contact_motion_cos,
    }
