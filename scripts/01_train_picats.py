#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os.path as osp
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', 'src')))

from pinn_hoi.data.arctic_io import ArcticPICATSWindowDataset
from pinn_hoi.losses.picats_losses import compute_losses, compute_metrics
from pinn_hoi.models.picats import build_model_from_config
from pinn_hoi.utils.io import (
    append_jsonl,
    detach_to_float_dict,
    ensure_dir,
    load_config,
    mean_float_dict,
    save_json,
    seed_everything,
    to_device,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--epochs', type=int, default=-1)
    p.add_argument('--overfit-batches', type=int, default=0)
    p.add_argument('--resume', default='')
    return p.parse_args()


def physics_scale_for_epoch(epoch: int, cfg: Dict) -> float:
    sch = cfg.get('schedule', {})
    sup = int(sch.get('supervised_warmup_epochs', 0))
    warm = int(sch.get('physics_warmup_epochs', 20))
    max_scale = float(sch.get('max_physics_scale', 1.0))
    if epoch < sup:
        return 0.0
    if warm <= 0:
        return max_scale
    x = min(max((epoch - sup + 1) / warm, 0.0), 1.0)
    # smooth ramp is more stable than linear for PINN residuals
    return max_scale * (0.5 - 0.5 * math.cos(math.pi * x))


def make_loader(cfg: Dict, split: str, overfit_batches: int = 0):
    list_path = cfg['train_list'] if split == 'train' else cfg['val_list']
    ds = ArcticPICATSWindowDataset(
        list_path=list_path,
        window=int(cfg['window']),
        stride=int(cfg['stride']) if split == 'train' else int(cfg['window']),
        contact_thresh_m=float(cfg.get('contact_thresh_m', 0.015)),
        preload=False,
    )
    if overfit_batches > 0:
        n = min(len(ds), int(cfg['batch_size']) * overfit_batches)
        ds = Subset(ds, list(range(n)))
    return DataLoader(
        ds,
        batch_size=int(cfg['batch_size']),
        shuffle=(split == 'train'),
        num_workers=int(cfg.get('num_workers', 4)),
        pin_memory=True,
        drop_last=(split == 'train' and len(ds) >= int(cfg['batch_size'])),
    )


def save_ckpt(path: str, model, optimizer, scaler, epoch: int, best_metric: float, cfg: Dict):
    ensure_dir(osp.dirname(path))
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_metric': best_metric,
        'config': cfg,
    }, path)


def load_ckpt(path: str, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    if optimizer is not None and 'optimizer' in ckpt and ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and ckpt.get('scaler') is not None:
        scaler.load_state_dict(ckpt['scaler'])
    return int(ckpt.get('epoch', -1)) + 1, float(ckpt.get('best_metric', 1e9))


def run_eval(model, loader, cfg, device):
    model.eval()
    metrics_all = []
    losses_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval', leave=False):
            batch = to_device(batch, device)
            out = model(batch)
            _, losses = compute_losses(batch, out, cfg, physics_scale=1.0)
            metrics = compute_metrics(batch, out)
            metrics_all.append(metrics)
            losses_all.append(losses)
    md = mean_float_dict(metrics_all)
    ld = {f'val_loss/{k}': v for k, v in mean_float_dict(losses_all).items()}
    md.update(ld)
    return md


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs > 0:
        cfg['epochs'] = args.epochs
    seed_everything(int(cfg.get('seed', 2026)))
    out_dir = ensure_dir(cfg['out_dir'])
    save_json(cfg, osp.join(out_dir, 'config.resolved.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = make_loader(cfg, 'train', args.overfit_batches)
    val_loader = make_loader(cfg, 'val', args.overfit_batches if args.overfit_batches > 0 else 0)
    if len(train_loader) == 0:
        raise RuntimeError('Empty training loader. Check train_list/window/stride.')

    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg.get('weight_decay', 0.0)))
    total_steps = max(1, int(cfg['epochs']) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=float(cfg.get('min_lr', 1e-5)),
    )
    use_amp = bool(cfg.get('use_amp', True)) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    start_epoch, best_metric = 0, 1e9
    if args.resume:
        start_epoch, best_metric = load_ckpt(args.resume, model, optimizer, scaler)
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    metric_path = osp.join(out_dir, 'metrics.jsonl')
    for epoch in range(start_epoch, int(cfg['epochs'])):
        model.train()
        pscale = physics_scale_for_epoch(epoch, cfg)
        train_losses = []
        pbar = tqdm(train_loader, desc=f'train epoch {epoch+1}/{cfg["epochs"]} p={pscale:.3f}')
        for batch in pbar:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                out = model(batch)
                total, losses = compute_losses(batch, out, cfg, physics_scale=pscale)
            scaler.scale(total).backward()
            if float(cfg.get('grad_clip_norm', 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip_norm']))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(losses)
            pbar.set_postfix({
                'loss': float(total.detach().cpu()),
                'trans': float(losses['obj_trans'].detach().cpu()),
                'c_bce': float(losses['contact_bce'].detach().cpu()),
            })

        train_log = {f'train_loss/{k}': v for k, v in mean_float_dict(train_losses).items()}
        val_log = run_eval(model, val_loader, cfg, device)
        log = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0], 'physics_scale': pscale}
        log.update(train_log)
        log.update(val_log)
        append_jsonl(log, metric_path)
        print(log)

        val_key = float(val_log.get('obj_trans_err_m', 1e9)) + 0.05 * float(val_log.get('obj_rot_err_rad', 1e9))
        save_ckpt(osp.join(out_dir, 'last.pt'), model, optimizer, scaler, epoch, best_metric, cfg)
        if val_key < best_metric:
            best_metric = val_key
            save_ckpt(osp.join(out_dir, 'best.pt'), model, optimizer, scaler, epoch, best_metric, cfg)
            print(f'[BEST] epoch={epoch+1} score={best_metric:.6f}')


if __name__ == '__main__':
    main()
