#!/usr/bin/env python
from __future__ import annotations

import argparse
import os.path as osp

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', 'src')))

from pinn_hoi.data.arctic_io import ArcticPICATSWindowDataset
from pinn_hoi.losses.picats_losses import compute_losses, compute_metrics
from pinn_hoi.models.picats import build_model_from_config
from pinn_hoi.utils.io import aggregate_eval_metrics, ensure_dir, load_config, mean_float_dict, save_json, to_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--out', default='')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    list_path = cfg[f'{args.split}_list'] if f'{args.split}_list' in cfg else cfg['val_list']
    ds = ArcticPICATSWindowDataset(list_path, int(cfg['window']), int(cfg['window']), float(cfg.get('contact_thresh_m', 0.015)))
    loader = DataLoader(ds, batch_size=int(cfg['batch_size']), shuffle=False, num_workers=int(cfg.get('num_workers', 4)), pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model_from_config(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    metrics_all, losses_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'eval {args.split}'):
            batch = to_device(batch, device)
            out = model(batch)
            _, losses = compute_losses(batch, out, cfg, physics_scale=1.0)
            metrics_all.append(compute_metrics(batch, out))
            losses_all.append(losses)
    result = aggregate_eval_metrics(metrics_all)
    result.update({f'loss/{k}': v for k, v in mean_float_dict(losses_all).items()})
    out_path = args.out or osp.join(cfg['out_dir'], f'eval_{args.split}.json')
    ensure_dir(osp.dirname(out_path))
    save_json(result, out_path)
    print(result)
    print(f'[OK] saved {out_path}')


if __name__ == '__main__':
    main()
