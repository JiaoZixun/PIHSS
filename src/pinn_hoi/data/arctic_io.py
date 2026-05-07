from __future__ import annotations

import os.path as osp
from glob import glob
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from pinn_hoi.utils.io import npz_to_dict, read_list


def _load_npy_auto(path: str):
    """Load npy robustly.

    ARCTIC raw files use two styles:
      - *.mano.npy / *.egocam.dist.npy / *.smplx.npy are pickled 0-d object dicts.
      - *.object.npy is a plain numeric [T,7] ndarray.

    Calling `.item()` blindly on the latter gives:
      ValueError: can only convert an array of size 1 to a Python scalar
    """
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        return arr.item()
    return arr


def _first_existing_key(d: Dict[str, Any], keys: List[str], required: bool = True, default=None):
    for k in keys:
        if k in d:
            return d[k]
    if required:
        raise KeyError(f'None of keys {keys} found in dict. Existing keys={list(d.keys())[:50]}')
    return default


def _to_np_float(x, name: str, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f'{name} expects ndim={ndim}, got shape={arr.shape}')
    return arr


def _repeat_shape_to_frames(shape: np.ndarray, T: int) -> np.ndarray:
    shape = np.asarray(shape, dtype=np.float32)
    if shape.ndim == 1:
        return np.repeat(shape[None], T, axis=0)
    if shape.ndim == 2 and shape.shape[0] == 1:
        return np.repeat(shape, T, axis=0)
    if shape.ndim == 2 and shape.shape[0] == T:
        return shape
    # Conservative fallback: ARCTIC shape is normally [10], constant per sequence.
    return np.repeat(shape.reshape(1, -1)[:, :10], T, axis=0)


def collect_raw_mano_files(arctic_root: str) -> List[str]:
    return sorted(glob(osp.join(arctic_root, 'raw_seqs', '*', '*.mano.npy')))


def infer_object_name_from_seq(seq_base: str) -> str:
    # ARCTIC sequence base examples: capsulemachine_use_01, box_grab_01.
    return seq_base.split('_')[0]


def _parse_official_mano(mano: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Parse official ARCTIC nested MANO dict.

    Official format:
      mano['right']['rot'], ['pose'], ['trans'], ['shape'], ['fitting_err']
      mano['left' ]['rot'], ['pose'], ['trans'], ['shape'], ['fitting_err']
    """
    r = mano['right']
    l = mano['left']
    rot_r = _to_np_float(r['rot'], 'right.rot', ndim=2)
    pose_r = _to_np_float(r['pose'], 'right.pose', ndim=2)
    trans_r = _to_np_float(r['trans'], 'right.trans', ndim=2)
    rot_l = _to_np_float(l['rot'], 'left.rot', ndim=2)
    pose_l = _to_np_float(l['pose'], 'left.pose', ndim=2)
    trans_l = _to_np_float(l['trans'], 'left.trans', ndim=2)
    T = rot_r.shape[0]
    return {
        'rot_r': rot_r.reshape(T, 3),
        'pose_r': pose_r.reshape(T, -1),
        'trans_r': trans_r.reshape(T, 3),
        'shape_r': _repeat_shape_to_frames(r['shape'], T),
        'rot_l': rot_l.reshape(T, 3),
        'pose_l': pose_l.reshape(T, -1),
        'trans_l': trans_l.reshape(T, 3),
        'shape_l': _repeat_shape_to_frames(l['shape'], T),
    }


def _parse_flat_mano(mano: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Parse fallback flattened variants used by some processed/custom files."""
    rot_r = _first_existing_key(mano, ['rot_r', 'right_rot', 'rot_right'])
    pose_r = _first_existing_key(mano, ['pose_r', 'right_pose', 'pose_right'])
    trans_r = _first_existing_key(mano, ['trans_r', 'right_trans', 'trans_right'])
    shape_r = _first_existing_key(mano, ['shape_r', 'right_shape', 'shape_right'])
    rot_l = _first_existing_key(mano, ['rot_l', 'left_rot', 'rot_left'])
    pose_l = _first_existing_key(mano, ['pose_l', 'left_pose', 'pose_left'])
    trans_l = _first_existing_key(mano, ['trans_l', 'left_trans', 'trans_left'])
    shape_l = _first_existing_key(mano, ['shape_l', 'left_shape', 'shape_left'])
    rot_r = _to_np_float(rot_r, 'rot_r', ndim=2)
    T = rot_r.shape[0]
    return {
        'rot_r': rot_r.reshape(T, 3),
        'pose_r': _to_np_float(pose_r, 'pose_r').reshape(T, -1),
        'trans_r': _to_np_float(trans_r, 'trans_r').reshape(T, 3),
        'shape_r': _repeat_shape_to_frames(shape_r, T),
        'rot_l': _to_np_float(rot_l, 'rot_l').reshape(T, 3),
        'pose_l': _to_np_float(pose_l, 'pose_l').reshape(T, -1),
        'trans_l': _to_np_float(trans_l, 'trans_l').reshape(T, 3),
        'shape_l': _repeat_shape_to_frames(shape_l, T),
    }


def _parse_object_pose7(obj_raw: Any, T: int) -> np.ndarray:
    """Parse official ARCTIC object pose.

    Official raw object file is numeric [T,7]:
      [obj_arti, obj_rot_axis_angle(3), obj_trans(3)]
    Some custom variants may be dict-like, so we keep a fallback.
    """
    if isinstance(obj_raw, dict):
        angle = _first_existing_key(obj_raw, ['angle', 'arti', 'articulation', 'obj_arti'], required=False)
        obj_rot = _first_existing_key(obj_raw, ['rot', 'global_orient', 'object_rot', 'obj_rot'])
        obj_trans = _first_existing_key(obj_raw, ['trans', 'transl', 'object_trans', 'obj_trans'])
        if angle is None:
            angle = np.zeros((T, 1), dtype=np.float32)
        angle = np.asarray(angle, dtype=np.float32).reshape(T, 1)
        obj_rot = np.asarray(obj_rot, dtype=np.float32).reshape(T, 3)
        obj_trans = np.asarray(obj_trans, dtype=np.float32).reshape(T, 3)
        pose7 = np.concatenate([angle, obj_rot, obj_trans], axis=-1)
    else:
        pose7 = np.asarray(obj_raw, dtype=np.float32)
        if pose7.ndim != 2 or pose7.shape[1] != 7:
            raise ValueError(f'Official ARCTIC object pose should be [T,7], got shape={pose7.shape}')
        if pose7.shape[0] != T:
            min_t = min(T, pose7.shape[0])
            pose7 = pose7[:min_t]
        pose7 = pose7.astype(np.float32)

    # Official ARCTIC object translation is in millimeters and official processing uses /1000.
    # Keep this heuristic to also tolerate already-meter custom files.
    obj_trans = pose7[:, 4:7]
    if np.nanmedian(np.linalg.norm(obj_trans, axis=-1)) > 5.0:
        pose7[:, 4:7] = obj_trans / 1000.0
    return pose7.astype(np.float32)


def _parse_egocam(ego_raw: Any, T: int):
    """Parse official *.egocam.dist.npy dict into world2ego/K/dist."""
    if isinstance(ego_raw, dict):
        if 'R_k_cam_np' in ego_raw and 'T_k_cam_np' in ego_raw:
            R = np.asarray(ego_raw['R_k_cam_np'], dtype=np.float32).reshape(-1, 3, 3)
            Tvec = np.asarray(ego_raw['T_k_cam_np'], dtype=np.float32).reshape(-1, 3)
            n = min(T, R.shape[0], Tvec.shape[0])
            world2ego = np.zeros((n, 4, 4), dtype=np.float32)
            world2ego[:, :3, :3] = R[:n]
            world2ego[:, :3, 3] = Tvec[:n]
            world2ego[:, 3, 3] = 1.0
            K_ego = np.asarray(ego_raw.get('intrinsics', np.eye(3)), dtype=np.float32)
            dist = np.asarray(ego_raw.get('dist8', np.zeros(8)), dtype=np.float32)
            return world2ego, K_ego, dist
        if 'world2ego' in ego_raw:
            world2ego = np.asarray(ego_raw['world2ego'], dtype=np.float32).reshape(-1, 4, 4)
            K_ego = np.asarray(ego_raw.get('K_ego', ego_raw.get('intrinsics', np.eye(3))), dtype=np.float32)
            dist = np.asarray(ego_raw.get('dist', ego_raw.get('dist8', np.zeros(8))), dtype=np.float32)
            return world2ego[:T], K_ego, dist
    return None, np.eye(3, dtype=np.float32), np.zeros((8,), dtype=np.float32)


def load_arctic_raw_bundle(mano_path: str) -> Dict[str, Any]:
    """Load ARCTIC raw .mano.npy + .object.npy + .egocam.dist.npy.

    Output convention:
      - MANO and object are kept in world coordinate.
      - obj_pose7 = [articulation, axis-angle rotation, translation_in_meter].
      - right/left MANO are returned as separate arrays.
    """
    mano = _load_npy_auto(mano_path)
    if not isinstance(mano, dict):
        raise ValueError(f'Expected MANO npy to be a dict, got type={type(mano)} path={mano_path}')

    obj_path = mano_path.replace('.mano.npy', '.object.npy')
    if not osp.exists(obj_path):
        raise FileNotFoundError(obj_path)
    obj_raw = _load_npy_auto(obj_path)

    seq_base = osp.basename(mano_path).replace('.mano.npy', '')
    sid = mano_path.split('/')[-2]
    obj_name = infer_object_name_from_seq(seq_base)

    if 'right' in mano and 'left' in mano:
        m = _parse_official_mano(mano)
    else:
        m = _parse_flat_mano(mano)
    T = int(m['rot_r'].shape[0])

    obj_pose7 = _parse_object_pose7(obj_raw, T)
    # Align all arrays to common T if needed.
    T = min(T, int(obj_pose7.shape[0]))
    for k in list(m.keys()):
        m[k] = m[k][:T].astype(np.float32)
    obj_pose7 = obj_pose7[:T].astype(np.float32)

    ego_p = mano_path.replace('.mano.npy', '.egocam.dist.npy')
    if osp.exists(ego_p):
        ego_raw = _load_npy_auto(ego_p)
        world2ego, K_ego, dist = _parse_egocam(ego_raw, T)
    else:
        world2ego, K_ego, dist = None, np.eye(3, dtype=np.float32), np.zeros((8,), dtype=np.float32)

    return {
        'sid': sid,
        'seq_base': seq_base,
        'seq_name': f'{sid}/{seq_base}',
        'obj_name': obj_name,
        'rot_r': m['rot_r'],
        'pose_r': m['pose_r'],
        'trans_r': m['trans_r'],
        'shape_r': m['shape_r'],
        'rot_l': m['rot_l'],
        'pose_l': m['pose_l'],
        'trans_l': m['trans_l'],
        'shape_l': m['shape_l'],
        'obj_pose7': obj_pose7,
        'dist': np.asarray(dist, dtype=np.float32),
        'world2ego': world2ego.astype(np.float32) if world2ego is not None else None,
        'K_ego': np.asarray(K_ego, dtype=np.float32),
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