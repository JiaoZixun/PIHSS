from __future__ import annotations

import os.path as osp
from glob import glob
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from pinn_hoi.utils.io import npz_to_dict, read_list


def _first_existing_key(d: Dict[str, Any], keys: List[str], required: bool = True, default=None):
    for k in keys:
        if k in d:
            return d[k]
    if required:
        raise KeyError(f'None of keys {keys} found in dict. Existing keys={list(d.keys())[:50]}')
    return default


def collect_raw_mano_files(arctic_root: str) -> List[str]:
    return sorted(glob(osp.join(arctic_root, 'raw_seqs', '*', '*.mano.npy')))


def infer_object_name_from_seq(seq_base: str) -> str:
    # ARCTIC sequence base examples: capsulemachine_use_01, box_grab_01.
    return seq_base.split('_')[0]


def load_arctic_raw_bundle(mano_path: str) -> Dict[str, Any]:
    """Load ARCTIC raw .mano.npy + .object.npy + optional egocam files.

    This reader is intentionally tolerant to small key-name variations. Object pose is converted to pose7:
      [articulation, axis-angle rotation, translation_in_meter]
    """
    mano = np.load(mano_path, allow_pickle=True).item()
    obj_path = mano_path.replace('.mano.npy', '.object.npy')
    if not osp.exists(obj_path):
        raise FileNotFoundError(obj_path)
    obj = np.load(obj_path, allow_pickle=True).item()

    seq_base = osp.basename(mano_path).replace('.mano.npy', '')
    sid = mano_path.split('/')[-2]
    obj_name = infer_object_name_from_seq(seq_base)

    rot_r = _first_existing_key(mano, ['rot_r', 'right_rot', 'rot_right'])
    pose_r = _first_existing_key(mano, ['pose_r', 'right_pose', 'pose_right'])
    trans_r = _first_existing_key(mano, ['trans_r', 'right_trans', 'trans_right'])
    shape_r = _first_existing_key(mano, ['shape_r', 'right_shape', 'shape_right'])

    rot_l = _first_existing_key(mano, ['rot_l', 'left_rot', 'rot_left'])
    pose_l = _first_existing_key(mano, ['pose_l', 'left_pose', 'pose_left'])
    trans_l = _first_existing_key(mano, ['trans_l', 'left_trans', 'trans_left'])
    shape_l = _first_existing_key(mano, ['shape_l', 'left_shape', 'shape_left'])

    angle = _first_existing_key(obj, ['angle', 'arti', 'articulation'], required=False)
    if angle is None:
        angle = np.zeros((np.asarray(rot_r).shape[0], 1), dtype=np.float32)
    angle = np.asarray(angle, dtype=np.float32).reshape(-1, 1)

    obj_rot = _first_existing_key(obj, ['rot', 'global_orient', 'object_rot', 'obj_rot'])
    obj_trans = _first_existing_key(obj, ['trans', 'transl', 'object_trans', 'obj_trans'])
    obj_rot = np.asarray(obj_rot, dtype=np.float32).reshape(angle.shape[0], 3)
    obj_trans = np.asarray(obj_trans, dtype=np.float32).reshape(angle.shape[0], 3)

    # ARCTIC raw object translation is commonly millimeter scale in official processing.
    # Keep MANO/object in meters by default when values look too large.
    if np.nanmedian(np.linalg.norm(obj_trans, axis=-1)) > 5.0:
        obj_trans = obj_trans / 1000.0

    obj_pose7 = np.concatenate([angle, obj_rot, obj_trans], axis=-1).astype(np.float32)

    dist_p = mano_path.replace('.mano.npy', '.egocam.dist.npy')
    world2ego_p = mano_path.replace('.mano.npy', '.egocam.world2ego.npy')
    intr_p = mano_path.replace('.mano.npy', '.egocam.intrinsics.npy')
    dist = np.load(dist_p).astype(np.float32) if osp.exists(dist_p) else np.zeros((8,), dtype=np.float32)
    world2ego = np.load(world2ego_p).astype(np.float32) if osp.exists(world2ego_p) else None
    K_ego = np.load(intr_p).astype(np.float32) if osp.exists(intr_p) else np.eye(3, dtype=np.float32)

    return {
        'sid': sid,
        'seq_base': seq_base,
        'seq_name': f'{sid}/{seq_base}',
        'obj_name': obj_name,
        'rot_r': np.asarray(rot_r, dtype=np.float32),
        'pose_r': np.asarray(pose_r, dtype=np.float32),
        'trans_r': np.asarray(trans_r, dtype=np.float32),
        'shape_r': np.asarray(shape_r, dtype=np.float32),
        'rot_l': np.asarray(rot_l, dtype=np.float32),
        'pose_l': np.asarray(pose_l, dtype=np.float32),
        'trans_l': np.asarray(trans_l, dtype=np.float32),
        'shape_l': np.asarray(shape_l, dtype=np.float32),
        'obj_pose7': obj_pose7,
        'dist': dist,
        'world2ego': world2ego,
        'K_ego': K_ego,
    }


class ArcticPICATSWindowDataset(Dataset):
    """Windowed dataset over processed PI-CATS npz sequences."""

    tensor_keys = [
        'hand_vertices', 'hand_joints', 'hand_endpoints',
        'mano_rot', 'mano_pose', 'mano_trans', 'mano_shape',
        'obj_pose7', 'obj_points_world', 'obj_points_canonical',
        'contact_label', 'endpoint_obj_min_dist', 'endpoint_nearest_obj_idx',
    ]

    def __init__(
        self,
        list_path: str,
        window: int,
        stride: int,
        contact_thresh_m: float = 0.015,
        max_windows: int = -1,
        preload: bool = False,
    ) -> None:
        self.files = read_list(list_path)
        self.window = int(window)
        self.stride = int(stride)
        self.contact_thresh_m = float(contact_thresh_m)
        self.index: List[tuple[str, int]] = []
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        for fp in self.files:
            with np.load(fp, allow_pickle=True) as data:
                T = int(data['obj_pose7'].shape[0])
            if T < self.window:
                continue
            for st in range(0, T - self.window + 1, self.stride):
                self.index.append((fp, st))
                if 0 < max_windows <= len(self.index):
                    break
            if 0 < max_windows <= len(self.index):
                break
        if preload:
            for fp in sorted(set(f for f, _ in self.index)):
                self.cache[fp] = npz_to_dict(fp)

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, fp: str) -> Dict[str, np.ndarray]:
        if fp in self.cache:
            return self.cache[fp]
        return npz_to_dict(fp)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        fp, st = self.index[idx]
        seq = self._load(fp)
        ed = st + self.window
        out: Dict[str, torch.Tensor | str] = {
            'seq_path': fp,
            'window_start': torch.tensor(st, dtype=torch.long),
            'contact_thresh_m': torch.tensor(self.contact_thresh_m, dtype=torch.float32),
        }
        for k in self.tensor_keys:
            if k not in seq:
                continue
            arr = seq[k]
            if k == 'obj_points_canonical':
                sliced = arr
            else:
                sliced = arr[st:ed]
            if k == 'endpoint_nearest_obj_idx':
                out[k] = torch.from_numpy(sliced).long()
            else:
                out[k] = torch.from_numpy(sliced).float()
        return out
