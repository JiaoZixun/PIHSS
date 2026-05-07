#!/usr/bin/env python
from __future__ import annotations

import argparse
import os.path as osp
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', 'src')))

from pinn_hoi.common.geometry_engine import UnifiedGeometryEngine, load_object_canonical_points
from pinn_hoi.common.rot import pose7_to_points
from pinn_hoi.data.arctic_io import collect_raw_mano_files, load_arctic_raw_bundle
from pinn_hoi.utils.io import ensure_dir, write_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arctic-root', required=True, help='Path to arctic_data/data')
    p.add_argument('--mano-root', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--split-out', required=True)
    p.add_argument('--num-object-points', type=int, default=2048)
    p.add_argument('--contact-thresh-m', type=float, default=0.015)
    p.add_argument('--object-verts-scale', type=float, default=1.0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seqs', type=int, default=-1)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--test-ratio', type=float, default=0.1)
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def build_one(mano_p: str, args, object_cache: Dict[str, np.ndarray]) -> str:
    bundle = load_arctic_raw_bundle(mano_p)
    sid, seq_base, obj_name = bundle['sid'], bundle['seq_base'], bundle['obj_name']
    out_p = osp.join(args.out_dir, sid, seq_base + '.npz')
    if osp.exists(out_p) and not args.force:
        return out_p

    if obj_name not in object_cache:
        object_cache[obj_name] = load_object_canonical_points(
            args.arctic_root,
            obj_name,
            num_points=args.num_object_points,
            object_verts_scale=args.object_verts_scale,
        )
    obj_c = object_cache[obj_name].astype(np.float32)

    geom = UnifiedGeometryEngine(
        mano_root=args.mano_root,
        device=args.device,
        contact_thresh_m=args.contact_thresh_m,
        object_points=torch.from_numpy(obj_c),
    )
    left = geom.decode_mano_np_sequence(bundle['rot_l'], bundle['pose_l'], bundle['trans_l'], bundle['shape_l'], is_right=False)
    right = geom.decode_mano_np_sequence(bundle['rot_r'], bundle['pose_r'], bundle['trans_r'], bundle['shape_r'], is_right=True)

    hand_vertices = torch.stack([left.vertices, right.vertices], dim=1)
    hand_joints = torch.stack([left.joints, right.joints], dim=1)
    hand_endpoints = torch.stack([left.endpoints, right.endpoints], dim=1)

    obj_pose7 = torch.from_numpy(bundle['obj_pose7']).to(geom.device)
    obj_points_world = pose7_to_points(torch.from_numpy(obj_c).to(geom.device), obj_pose7)
    contact_label, min_dist, nearest_idx = geom.compute_contact_labels(hand_endpoints, obj_points_world)

    ensure_dir(osp.dirname(out_p))
    np.savez_compressed(
        out_p,
        seq_name=np.array(bundle['seq_name']),
        obj_name=np.array(obj_name),
        mano_rot=np.stack([bundle['rot_l'], bundle['rot_r']], axis=1).astype(np.float32),
        mano_pose=np.stack([bundle['pose_l'], bundle['pose_r']], axis=1).astype(np.float32),
        mano_trans=np.stack([bundle['trans_l'], bundle['trans_r']], axis=1).astype(np.float32),
        mano_shape=np.stack([bundle['shape_l'], bundle['shape_r']], axis=1).astype(np.float32),
        hand_vertices=hand_vertices.cpu().numpy().astype(np.float32),
        hand_joints=hand_joints.cpu().numpy().astype(np.float32),
        hand_endpoints=hand_endpoints.cpu().numpy().astype(np.float32),
        obj_pose7=bundle['obj_pose7'].astype(np.float32),
        obj_points_canonical=obj_c.astype(np.float32),
        obj_points_world=obj_points_world.cpu().numpy().astype(np.float32),
        contact_label=contact_label.cpu().numpy().astype(np.float32),
        endpoint_obj_min_dist=min_dist.cpu().numpy().astype(np.float32),
        endpoint_nearest_obj_idx=nearest_idx.cpu().numpy().astype(np.int64),
        world2ego=bundle['world2ego'] if bundle['world2ego'] is not None else np.zeros((1, 4, 4), dtype=np.float32),
        K_ego=bundle['K_ego'].astype(np.float32),
        dist=bundle['dist'].astype(np.float32),
    )
    return out_p


def split_files(files: List[str], val_ratio: float, test_ratio: float):
    files = sorted(files)
    n = len(files)
    n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
    test = files[:n_test]
    val = files[n_test:n_test + n_val]
    train = files[n_test + n_val:]
    if not train:
        train = files
    if not val:
        val = files[:1]
    if not test:
        test = files[:1]
    return train, val, test


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.split_out)
    mano_files = collect_raw_mano_files(args.arctic_root)
    if args.max_seqs > 0:
        mano_files = mano_files[:args.max_seqs]
    object_cache: Dict[str, np.ndarray] = {}
    out_files: List[str] = []
    for mp in tqdm(mano_files, desc='build ARCTIC PI-CATS sequences'):
        try:
            out_files.append(build_one(mp, args, object_cache))
        except Exception as e:
            print(f'[WARN] failed {mp}: {repr(e)}')
    train, val, test = split_files(out_files, args.val_ratio, args.test_ratio)
    write_list(train, osp.join(args.split_out, 'train.txt'))
    write_list(val, osp.join(args.split_out, 'val.txt'))
    write_list(test, osp.join(args.split_out, 'test.txt'))
    print(f'[OK] sequences={len(out_files)} train={len(train)} val={len(val)} test={len(test)}')


if __name__ == '__main__':
    main()
