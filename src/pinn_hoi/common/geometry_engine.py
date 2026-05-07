from __future__ import annotations

import os.path as osp
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from pinn_hoi.common.rot import pose7_to_points
from pinn_hoi.utils.io import patch_numpy_legacy_aliases


# MANO fingertip vertex ids used by common MANO/manopth pipelines.
MANO_TIP_VERTEX_IDS = torch.tensor([745, 317, 444, 556, 673], dtype=torch.long)
# Convert MANO 16 joints + 5 tips to common 21-joint order:
# wrist, thumb(3+tip), index(3+tip), middle(3+tip), ring(3+tip), pinky(3+tip)
MANO_21_ORDER = torch.tensor([
    0, 13, 14, 15, 16,
    1, 2, 3, 17,
    4, 5, 6, 18,
    10, 11, 12, 19,
    7, 8, 9, 20,
], dtype=torch.long)


@dataclass
class ManoDecodeResult:
    vertices: torch.Tensor  # [T,778,3]
    joints: torch.Tensor    # [T,21,3]
    endpoints: torch.Tensor # [T,21,3]


class UnifiedGeometryEngine:
    """Centralized geometry engine.

    This is the only file that should decode MANO, transform object points, and compute hand-object contact. Train/eval/vis code must consume the stored outputs from this engine instead of reimplementing geometry.
    """

    def __init__(
        self,
        mano_root: str,
        device: str | torch.device = 'cuda',
        contact_thresh_m: float = 0.015,
        object_points: Optional[torch.Tensor] = None,
    ) -> None:
        patch_numpy_legacy_aliases()
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == 'cpu' else 'cpu')
        self.contact_thresh_m = float(contact_thresh_m)
        self.object_points = object_points.to(self.device).float() if object_points is not None else None
        self.mano_right = self._create_mano(mano_root, is_right=True)
        self.mano_left = self._create_mano(mano_root, is_right=False)

    def _create_mano(self, mano_root: str, is_right: bool):
        try:
            import smplx
            layer = smplx.create(
                model_path=mano_root,
                model_type='mano',
                is_rhand=is_right,
                use_pca=False,
                flat_hand_mean=False,
                batch_size=1,
            ).to(self.device)
            layer.eval()
            return layer
        except Exception as e:
            raise RuntimeError(
                'Failed to create MANO layer. Ensure smplx is installed and mano_root contains MANO models. '
                f'mano_root={mano_root}, is_right={is_right}, error={repr(e)}'
            )

    @staticmethod
    def build_21_joints(vertices: torch.Tensor, mano_joints: torch.Tensor) -> torch.Tensor:
        """Build stable 21 endpoints from MANO vertices and joints.

        Supports MANO layers that return either 16 joints or >=21 joints. For consistency across all stages, we always rebuild the 21-joint layout from 16 joints + tip vertices when possible.
        """
        T = vertices.shape[0]
        tip_ids = MANO_TIP_VERTEX_IDS.to(vertices.device)
        order = MANO_21_ORDER.to(vertices.device)
        if mano_joints.shape[1] >= 16:
            base16 = mano_joints[:, :16]
            tips = vertices[:, tip_ids]
            j21_raw = torch.cat([base16, tips], dim=1)  # [T,21,3]
            return j21_raw[:, order]
        if mano_joints.shape[1] == 21:
            return mano_joints
        raise ValueError(f'Unexpected MANO joints shape: {mano_joints.shape}')

    @torch.no_grad()
    def decode_mano_np_sequence(
        self,
        global_orient: np.ndarray,
        hand_pose: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray,
        is_right: bool,
        batch_size: int = 256,
    ) -> ManoDecodeResult:
        layer = self.mano_right if is_right else self.mano_left
        rot = torch.as_tensor(global_orient, dtype=torch.float32, device=self.device).reshape(-1, 3)
        pose = torch.as_tensor(hand_pose, dtype=torch.float32, device=self.device).reshape(rot.shape[0], -1)
        trans = torch.as_tensor(transl, dtype=torch.float32, device=self.device).reshape(rot.shape[0], 3)
        b = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
        if b.dim() == 1:
            b = b[None].expand(rot.shape[0], -1)
        elif b.shape[0] == 1:
            b = b.expand(rot.shape[0], -1)
        elif b.shape[0] != rot.shape[0]:
            b = b[:1].expand(rot.shape[0], -1)

        verts_all, joints_all = [], []
        for start in range(0, rot.shape[0], batch_size):
            end = min(start + batch_size, rot.shape[0])
            out = layer(
                global_orient=rot[start:end],
                hand_pose=pose[start:end],
                betas=b[start:end],
                transl=trans[start:end],
                return_verts=True,
            )
            verts = out.vertices
            joints = self.build_21_joints(verts, out.joints)
            verts_all.append(verts)
            joints_all.append(joints)
        vertices = torch.cat(verts_all, dim=0)
        joints = torch.cat(joints_all, dim=0)
        return ManoDecodeResult(vertices=vertices, joints=joints, endpoints=joints)

    def object_points_world(self, obj_pose7: torch.Tensor, object_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        pts = object_points if object_points is not None else self.object_points
        if pts is None:
            raise ValueError('object_points must be provided either at init or call time.')
        return pose7_to_points(pts.to(obj_pose7.device), obj_pose7)

    @staticmethod
    def cdist_nearest_chunked(hand_ep: torch.Tensor, obj_pts: torch.Tensor, chunk: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest object point for endpoints.

        hand_ep: [...,E,3], obj_pts: [...,N,3]
        returns: min_dist [...,E], nearest_idx [...,E]
        """
        orig_shape = hand_ep.shape[:-2]
        E = hand_ep.shape[-2]
        N = obj_pts.shape[-2]
        h = hand_ep.reshape(-1, E, 3)
        o = obj_pts.reshape(-1, N, 3)
        min_ds, min_is = [], []
        for st in range(0, h.shape[0], chunk):
            ed = min(st + chunk, h.shape[0])
            d = torch.cdist(h[st:ed], o[st:ed])
            md, mi = d.min(dim=-1)
            min_ds.append(md)
            min_is.append(mi)
        min_d = torch.cat(min_ds, dim=0).reshape(*orig_shape, E)
        min_i = torch.cat(min_is, dim=0).reshape(*orig_shape, E)
        return min_d, min_i

    def compute_contact_labels(
        self,
        hand_endpoints: torch.Tensor,
        obj_points_world: torch.Tensor,
        thresh_m: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute endpoint contact labels.

        hand_endpoints: [T,2,21,3] or [B,T,2,21,3]
        obj_points_world: [T,N,3] or [B,T,N,3]
        """
        thresh = self.contact_thresh_m if thresh_m is None else float(thresh_m)
        if hand_endpoints.dim() == 4:
            T = hand_endpoints.shape[0]
            hand_flat = hand_endpoints.reshape(T, -1, 3)
            min_d, idx = self.cdist_nearest_chunked(hand_flat, obj_points_world)
            min_d = min_d.reshape(T, 2, 21)
            idx = idx.reshape(T, 2, 21)
        elif hand_endpoints.dim() == 5:
            B, T = hand_endpoints.shape[:2]
            hand_flat = hand_endpoints.reshape(B, T, -1, 3)
            min_d, idx = self.cdist_nearest_chunked(hand_flat, obj_points_world)
            min_d = min_d.reshape(B, T, 2, 21)
            idx = idx.reshape(B, T, 2, 21)
        else:
            raise ValueError(f'Unsupported endpoint shape: {hand_endpoints.shape}')
        label = (min_d <= thresh).float()
        return label, min_d, idx.long()


def farthest_point_sample_np(points: np.ndarray, npoint: int, random_start: bool = False) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    if N <= npoint:
        if N == npoint:
            return points
        pad_idx = np.random.choice(N, npoint - N, replace=True)
        return np.concatenate([points, points[pad_idx]], axis=0)
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((N,), dtype=np.float64) * 1e10
    farthest = np.random.randint(0, N) if random_start else 0
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest][None]
        dist = ((points - centroid) ** 2).sum(-1)
        distance = np.minimum(distance, dist)
        farthest = int(distance.argmax())
    return points[centroids]


def load_object_canonical_points(
    arctic_root: str,
    obj_name: str,
    num_points: int = 2048,
    object_verts_scale: float = 1.0,
) -> np.ndarray:
    """Load canonical object point cloud from ARCTIC object template mesh."""
    candidates = [
        osp.join(arctic_root, 'meta', 'object_vtemplates', obj_name, 'mesh.obj'),
        osp.join(arctic_root, 'meta', 'object_vtemplates', obj_name, f'{obj_name}.obj'),
        osp.join(arctic_root, 'meta', 'object_vtemplates', obj_name + '.obj'),
    ]
    mesh_p = next((p for p in candidates if osp.exists(p)), None)
    if mesh_p is None:
        raise FileNotFoundError(f'Cannot find object mesh for {obj_name}. Tried: {candidates}')
    mesh = trimesh.load(mesh_p, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32) * float(object_verts_scale)
    if verts.shape[0] > num_points:
        verts = farthest_point_sample_np(verts, num_points, random_start=False)
    elif verts.shape[0] < num_points:
        idx = np.random.choice(verts.shape[0], num_points - verts.shape[0], replace=True)
        verts = np.concatenate([verts, verts[idx]], axis=0)
    return verts.astype(np.float32)
