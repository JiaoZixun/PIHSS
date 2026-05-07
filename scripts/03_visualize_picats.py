#!/usr/bin/env python
from __future__ import annotations

import argparse
import os.path as osp

import numpy as np
import torch

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', 'src')))

from pinn_hoi.models.picats import build_model_from_config
from pinn_hoi.utils.io import load_config, npz_to_dict, to_device
from pinn_hoi.utils.vis import render_hand_object_prediction_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--seq', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-frames', type=int, default=120)
    return p.parse_args()


def make_batch(seq, window: int, max_frames: int):
    T = min(int(seq['obj_pose7'].shape[0]), max(window, max_frames))
    T = min(T, window)  # visualize first window for deterministic one-step predictions
    keys = [
        'hand_vertices', 'hand_joints', 'hand_endpoints', 'mano_rot', 'mano_pose', 'mano_trans', 'mano_shape',
        'obj_pose7', 'obj_points_world', 'obj_points_canonical', 'contact_label', 'endpoint_obj_min_dist', 'endpoint_nearest_obj_idx',
    ]
    batch = {}
    for k in keys:
        if k not in seq:
            continue
        arr = seq[k]
        if k == 'obj_points_canonical':
            batch[k] = torch.from_numpy(arr[None]).float()
        elif k == 'endpoint_nearest_obj_idx':
            batch[k] = torch.from_numpy(arr[:T][None]).long()
        else:
            batch[k] = torch.from_numpy(arr[:T][None]).float()
    return batch, T


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seq = npz_to_dict(args.seq)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model_from_config(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    batch, T = make_batch(seq, int(cfg['window']), int(args.max_frames))
    with torch.no_grad():
        out = model(to_device(batch, device))
    pred = seq['obj_pose7'][:T].copy()
    pred[1:T] = out['pred_next_pose7'][0, :T-1].detach().cpu().numpy()
    render_hand_object_prediction_video(
        hand_endpoints=seq['hand_endpoints'][:T],
        obj_points_world_gt=seq['obj_points_world'][:T],
        obj_points_canonical=seq['obj_points_canonical'],
        obj_pose7_gt=seq['obj_pose7'][:T],
        pred_pose7=pred,
        contact_label=seq.get('contact_label', None)[:T] if 'contact_label' in seq else None,
        out_path=args.out,
        max_frames=T,
    )
    print(f'[OK] saved {args.out}')


if __name__ == '__main__':
    main()
