#!/usr/bin/env python
from __future__ import annotations

import argparse
import os.path as osp
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', 'src')))

from pinn_hoi.data.arctic_io import ArcticPICATSWindowDataset
from pinn_hoi.losses.picats_losses import build_contact_target, compute_losses, compute_metrics
from pinn_hoi.models.picats import build_model_from_config
from pinn_hoi.common.finite_check import assert_finite_dict, assert_finite_tensor, check_model_grads_finite, check_model_params_finite
from pinn_hoi.utils.io import aggregate_eval_metrics, append_jsonl, ensure_dir, load_config, mean_float_dict, save_json, seed_everything, to_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--epochs', type=int, default=-1)
    p.add_argument('--overfit-batches', type=int, default=0)
    p.add_argument('--resume', default='')
    return p.parse_args()


def physics_scale_for_epoch(epoch: int, cfg: Dict) -> float:
    sch = cfg.get('physics_scale_schedule', cfg.get('schedule', {}))
    if sch.get('type', '') == 'linear':
        epochs = max(1, int(sch.get('epochs', 1)))
        start = float(sch.get('start', 0.0))
        end = float(sch.get('end', 1.0))
        alpha = min(max(epoch / max(epochs - 1, 1), 0.0), 1.0)
        return start + alpha * (end - start)
    return float(cfg.get('physics_scale', 1.0))


def build_dataset(cfg: Dict, split: str):
    list_path = cfg['train_list'] if split == 'train' else cfg['val_list']
    return ArcticPICATSWindowDataset(list_path=list_path, window=int(cfg['window']), stride=int(cfg['stride']) if split == 'train' else int(cfg['window']), contact_thresh_m=float(cfg.get('contact_thresh_m', 0.015)), preload=False)


def select_overfit_indices(ds, cfg: Dict, fallback_batches: int = 0):
    over = cfg.get('overfit', {})
    if not over.get('enabled', False) and fallback_batches <= 0:
        return None, None
    seed = int(over.get('fixed_indices_seed', cfg.get('seed', 2026)))
    g = torch.Generator().manual_seed(seed)
    n = int(over.get('num_windows', 0))
    if n <= 0:
        nb = int(over.get('num_batches', fallback_batches))
        n = int(cfg['batch_size']) * max(nb, 1)
    n = min(len(ds), n)
    candidates = list(range(len(ds)))
    stats = []
    for i in candidates:
        sample = ds[i]
        gt, valid = build_contact_target({'contact_label': sample['contact_label'].unsqueeze(0)})
        valid_count = int(valid.sum().item())
        pos_count = int((gt * valid).sum().item())
        ratio = float(pos_count / max(valid_count, 1))
        window_start = int(sample['window_start'].item())
        window_end = int(sample['window_end'].item()) if 'window_end' in sample else window_start + int(cfg['window'])
        stats.append({
            'sample_idx': i,
            'sequence_path': sample['seq_path'],
            'window_start': window_start,
            'window_end': window_end,
            'contact_valid_count': valid_count,
            'gt_contact_pos_count': pos_count,
            'gt_contact_ratio': ratio,
        })
    if over.get('enabled', False):
        min_valid = int(over.get('min_contact_valid_count', 1))
        min_pos = int(over.get('min_contact_pos_count', 20))
        min_ratio = float(over.get('min_contact_ratio', 0.03))
        max_ratio = float(over.get('max_contact_ratio', 0.6))
        filtered = [s for s in stats if s['contact_valid_count'] >= min_valid and s['gt_contact_pos_count'] >= min_pos and min_ratio <= s['gt_contact_ratio'] <= max_ratio]
        if len(filtered) == 0:
            pos_counts = torch.tensor([s['gt_contact_pos_count'] for s in stats], dtype=torch.float32)
            ratios = torch.tensor([s['gt_contact_ratio'] for s in stats], dtype=torch.float32)
            top = sorted(stats, key=lambda x: x['gt_contact_ratio'], reverse=True)[:20]
            top_lines = "\n".join([f"  {s['sequence_path']}::[{s['window_start']},{s['window_end']}) ratio={s['gt_contact_ratio']:.6f} pos={s['gt_contact_pos_count']} valid={s['contact_valid_count']}" for s in top])
            raise RuntimeError(
                "No positive contact samples found for overfit subset.\n"
                f"dataset_total_samples={len(ds)}\n"
                f"gt_contact_pos_count_max={pos_counts.max().item():.1f}, mean={pos_counts.mean().item():.3f}\n"
                f"gt_contact_ratio_max={ratios.max().item():.6f}, mean={ratios.mean().item():.6f}\n"
                f"top20_contact_ratio_samples=\n{top_lines}\n"
                f"filters: min_valid={min_valid}, min_pos={min_pos}, min_ratio={min_ratio}, max_ratio={max_ratio}"
            )
        candidates = [s['sample_idx'] for s in filtered]
    if bool(over.get('shuffle', False)):
        perm = torch.randperm(len(candidates), generator=g)[:n].tolist()
        idx = [candidates[j] for j in perm]
    else:
        idx = candidates[:n]
    return idx, {'all_windows': stats, 'selected_windows': [stats[i] for i in idx]}


def make_loader_from_dataset(ds, cfg: Dict, split: str, subset_indices=None, shuffle=None):
    if subset_indices is not None:
        ds = Subset(ds, subset_indices)
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(ds, batch_size=int(cfg['batch_size']), shuffle=shuffle, num_workers=int(cfg.get('num_workers', 4)), pin_memory=True, drop_last=(split == 'train' and len(ds) >= int(cfg['batch_size'])))


def save_ckpt(path: str, model, optimizer, scaler, epoch: int, scores: Dict[str, float], cfg: Dict):
    ensure_dir(osp.dirname(path))
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict() if scaler is not None else None, 'epoch': epoch, 'scores': scores, 'config': cfg}, path)


def load_ckpt(path: str, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and ckpt.get('scaler') is not None:
        scaler.load_state_dict(ckpt['scaler'])
    return int(ckpt.get('epoch', -1)) + 1


def run_eval(model, loader, cfg, device, split_tag: str, current_pscale: float, dump_dir: str):
    model.eval()
    metrics_all, losses_cur, losses_full = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc=f'eval {split_tag}', leave=False)):
            batch = to_device(batch, device)
            dctx = {'epoch': -1, 'batch_idx': bi, 'split': split_tag, 'dump_dir': dump_dir}
            assert_finite_dict('batch', batch, dctx)
            out = model(batch)
            assert_finite_dict('model_out', out, dctx)
            _, lc = compute_losses(batch, out, cfg, physics_scale=current_pscale, full_physics_scale=1.0)
            _, lf = compute_losses(batch, out, cfg, physics_scale=1.0, full_physics_scale=1.0)
            metrics_all.append(compute_metrics(batch, out))
            losses_cur.append(lc)
            losses_full.append(lf)
    md = {f'{split_tag}/{k}': v for k, v in aggregate_eval_metrics(metrics_all).items()}
    lcur = mean_float_dict(losses_cur)
    lfull = mean_float_dict(losses_full)
    out = {
        f'{split_tag}/total_current_scale': lcur['total_current_scale'],
        f'{split_tag}/total_full_physics': lfull['total_current_scale'],
        f'{split_tag}/supervised_total': lcur['supervised_total'],
        f'{split_tag}/physics_total_full': lfull['physics_total'],
        f'{split_tag}/physics_scale_current': lcur['physics_scale'],
        f'{split_tag}/physics_scale_full': lfull['physics_scale'],
    }
    for k, v in lcur.items():
        out[f'{split_tag}/loss_{k}'] = v
    md.update(out)
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
    train_ds, val_ds = build_dataset(cfg, 'train'), build_dataset(cfg, 'val')
    overfit_idx, overfit_stats = select_overfit_indices(train_ds, cfg, args.overfit_batches)
    overfit_same = bool(cfg.get('overfit', {}).get('same_val', False))
    train_loader = make_loader_from_dataset(train_ds, cfg, 'train', overfit_idx, shuffle=bool(cfg.get('overfit', {}).get('shuffle', False) if overfit_idx is not None else True))
    val_overfit_same_loader = make_loader_from_dataset(val_ds if not overfit_same else train_ds, cfg, 'val', overfit_idx if overfit_same else None, shuffle=False)
    val_regular_loader = make_loader_from_dataset(val_ds, cfg, 'val', None, shuffle=False)
    if overfit_idx is not None:
        selected = overfit_stats['selected_windows']
        ratios = torch.tensor([s['gt_contact_ratio'] for s in selected], dtype=torch.float32)
        summary = {
            'overfit_num_windows': len(selected),
            'gt_contact_ratio_mean': float(ratios.mean().item()) if len(selected) else 0.0,
            'gt_contact_ratio_min': float(ratios.min().item()) if len(selected) else 0.0,
            'gt_contact_ratio_max': float(ratios.max().item()) if len(selected) else 0.0,
            'total_positive_contact_count': int(sum(s['gt_contact_pos_count'] for s in selected)),
            'total_valid_contact_count': int(sum(s['contact_valid_count'] for s in selected)),
            'selected_windows': selected,
        }
        print(f'[overfit subset summary] {summary}')
        save_json(summary, osp.join(out_dir, 'overfit_subset_stats.json'))

    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg.get('weight_decay', 0.0)))
    total_steps = max(1, int(cfg['epochs']) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(cfg.get('min_lr', 1e-5)))
    use_amp = bool(cfg.get('use_amp', True)) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    start_epoch = load_ckpt(args.resume, model, optimizer, scaler) if args.resume else 0

    best = {'object': 1e9, 'contact': -1e9, 'balanced': 1e9}
    metric_path = osp.join(out_dir, 'metrics.jsonl')
    for epoch in range(start_epoch, int(cfg['epochs'])):
        model.train()
        pscale = physics_scale_for_epoch(epoch, cfg)
        train_losses = []
        pbar = tqdm(train_loader, desc=f'train {epoch+1}/{cfg["epochs"]} p={pscale:.3f}')
        for bi, batch in enumerate(pbar):
            batch = to_device(batch, device)
            dctx = {'epoch': epoch + 1, 'batch_idx': bi, 'split': 'train', 'dump_dir': osp.join(out_dir, 'debug_nan')}
            assert_finite_dict('batch', batch, dctx)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                out = model(batch)
                assert_finite_dict('model_out', out, dctx)
                total, losses = compute_losses(batch, out, cfg, physics_scale=pscale, full_physics_scale=1.0)
                assert_finite_dict('loss', losses, dctx)
                assert_finite_tensor('loss.total', total, dctx)
                metrics = compute_metrics(batch, out)
            scaler.scale(total).backward()
            grad_norm = check_model_grads_finite(model, dctx)
            scaler.step(optimizer)
            scaler.update()
            check_model_params_finite(model, dctx)
            scheduler.step()
            losses['grad_norm'] = torch.tensor(grad_norm, device=total.device)
            for k, v in metrics.items():
                if k.startswith('gt_contact_') or k.startswith('contact_valid_count') or k.startswith('pred_contact_prob') or k.startswith('pred_contact_pos_ratio@'):
                    losses[k] = v
            train_losses.append(losses)

        train_log = {f'train_overfit/{k}': v for k, v in mean_float_dict(train_losses).items()}
        train_log['train_loss/total_current_scale'] = train_log['train_overfit/total_current_scale']
        train_log['train_loss/physics_scale'] = train_log['train_overfit/physics_scale']
        val_same = run_eval(model, val_overfit_same_loader, cfg, device, 'val_overfit_same', pscale, osp.join(out_dir, 'debug_nan'))
        val_reg = run_eval(model, val_regular_loader, cfg, device, 'val_regular', pscale, osp.join(out_dir, 'debug_nan'))

        obj_score = float(val_same.get('val_overfit_same/obj_trans_err_m', 1e9)) + 0.1 * float(val_same.get('val_overfit_same/obj_rot_err_rad', 1e9)) + 0.05 * float(val_same.get('val_overfit_same/obj_arti_err', 1e9))
        contact_score = float(val_same.get('val_overfit_same/contact_f1', float('nan')))
        balanced = obj_score

        log = {'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0], 'object_score': obj_score, 'contact_score': contact_score, 'balanced_score': balanced, 'pred_rot_repr_type': 'axis_angle', 'gt_rot_repr_type': 'axis_angle'}
        log.update(train_log)
        log.update(val_same)
        log.update(val_reg)
        if not (torch.isfinite(torch.tensor(obj_score)) and torch.isfinite(torch.tensor(balanced))):
            raise FloatingPointError(f'Non-finite score at epoch {epoch+1}: {obj_score}, {balanced}')
        append_jsonl(log, metric_path)

        scores = {'object_score': obj_score, 'contact_score': contact_score, 'balanced_score': balanced}
        save_ckpt(osp.join(out_dir, 'last.pt'), model, optimizer, scaler, epoch, scores, cfg)
        save_ckpt(osp.join(out_dir, 'last_finite.pt'), model, optimizer, scaler, epoch, scores, cfg)
        if torch.isfinite(torch.tensor(obj_score)) and obj_score < best['object']:
            best['object'] = obj_score
            save_ckpt(osp.join(out_dir, 'best_object.pt'), model, optimizer, scaler, epoch, scores, cfg)
        if torch.isfinite(torch.tensor(contact_score)) and contact_score > best['contact']:
            best['contact'] = contact_score
            save_ckpt(osp.join(out_dir, 'best_contact.pt'), model, optimizer, scaler, epoch, scores, cfg)
        if torch.isfinite(torch.tensor(balanced)) and balanced < best['balanced']:
            best['balanced'] = balanced
            save_ckpt(osp.join(out_dir, 'best_balanced.pt'), model, optimizer, scaler, epoch, scores, cfg)


if __name__ == '__main__':
    main()
