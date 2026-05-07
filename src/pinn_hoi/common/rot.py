from __future__ import annotations

import torch
import torch.nn.functional as F


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula. axis_angle: [...,3], return [...,3,3]."""
    aa = axis_angle
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], dim=-1).reshape(*aa.shape[:-1], 3, 3)
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(*aa.shape[:-1], 3, 3)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    return I + sin * K + (1.0 - cos) * (K @ K)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Stable-ish matrix to axis-angle. R: [...,3,3]."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos)
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)
    axis = F.normalize(axis, dim=-1, eps=1e-8)
    return axis * angle.unsqueeze(-1)


def compose_axis_angle_delta(base_aa: torch.Tensor, delta_aa: torch.Tensor) -> torch.Tensor:
    """Apply delta rotation after base rotation and return axis-angle."""
    Rb = axis_angle_to_matrix(base_aa)
    Rd = axis_angle_to_matrix(delta_aa)
    return matrix_to_axis_angle(Rd @ Rb)


def pose7_to_points(points: torch.Tensor, pose7: torch.Tensor) -> torch.Tensor:
    """Transform canonical object points by pose7 [arti, axis-angle, trans].

    points: [N,3] or [B,N,3] or [B,T,N,3]
    pose7:  [B,7] or [B,T,7]
    """
    R = axis_angle_to_matrix(pose7[..., 1:4])
    t = pose7[..., 4:7]
    if points.dim() == 2:
        # [...,3,3] x [N,3]
        return torch.einsum('...ij,nj->...ni', R, points) + t.unsqueeze(-2)
    if points.dim() == pose7.dim():
        return torch.einsum('...ij,...nj->...ni', R, points) + t.unsqueeze(-2)
    if points.dim() == pose7.dim() + 1:
        return torch.einsum('...ij,...nj->...ni', R, points) + t.unsqueeze(-2)
    raise ValueError(f'Unsupported points dim={points.shape}, pose7 dim={pose7.shape}')


def gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by idx along point dimension.

    points: [B,T,N,3], idx: [B,T,2,21] -> [B,T,2,21,3]
    """
    B, T, N, C = points.shape
    flat_idx = idx.reshape(B, T, -1)
    gather_idx = flat_idx.unsqueeze(-1).expand(-1, -1, -1, C)
    gathered = torch.gather(points, 2, gather_idx)
    return gathered.reshape(*idx.shape, C)


def rotation_error_rad(pred_aa: torch.Tensor, gt_aa: torch.Tensor) -> torch.Tensor:
    Rp = axis_angle_to_matrix(pred_aa)
    Rg = axis_angle_to_matrix(gt_aa)
    R = Rp.transpose(-1, -2) @ Rg
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)
