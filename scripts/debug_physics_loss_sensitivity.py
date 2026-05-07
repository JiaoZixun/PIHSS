#!/usr/bin/env python
from __future__ import annotations
import argparse, os.path as osp, sys, torch
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', 'src')))
from torch.utils.data import DataLoader
from pinn_hoi.data.arctic_io import ArcticPICATSWindowDataset
from pinn_hoi.models.picats import build_model_from_config
from pinn_hoi.losses.picats_losses import compute_losses
from pinn_hoi.utils.io import load_config, to_device


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--ckpt', default=''); p.add_argument('--eps', type=float, default=1e-3); args = p.parse_args()
    cfg = load_config(args.config)
    ds = ArcticPICATSWindowDataset(cfg['train_list'], int(cfg['window']), int(cfg['window']), float(cfg.get('contact_thresh_m', 0.015)))
    loader = DataLoader(ds, batch_size=int(cfg['batch_size']), shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model_from_config(cfg).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
    model.train()
    batch = to_device(next(iter(loader)), device)
    out = model(batch)
    _, losses = compute_losses(batch, out, cfg, physics_scale=1.0)
    keys = ['no_contact_static','contact_velocity','contact_motion_align','impulse_dyn','angular_dyn','friction_cone','non_contact_force','delta_smooth','force_smooth']
    for k in keys:
        model.zero_grad(set_to_none=True)
        v = losses[k]
        v.backward(retain_graph=True)
        gn = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gn += float(p.grad.norm().detach().cpu())
        print({'loss':k,'value':float(v.detach().cpu()),'requires_grad':bool(v.requires_grad),'grad_norm':gn})
    with torch.no_grad():
        base = {k: float(losses[k].detach().cpu()) for k in keys}
        out2 = dict(out)
        out2['delta_pose7'] = out['delta_pose7'] + args.eps
        out2['pred_next_pose7'] = out['pred_next_pose7'] + torch.cat([torch.zeros_like(out['pred_next_pose7'][..., :4]), torch.full_like(out['pred_next_pose7'][..., 4:7], args.eps)], dim=-1)
        out2['contact_logits'] = out['contact_logits'] + args.eps
        out2['contact_prob'] = torch.sigmoid(out2['contact_logits'])
        _, losses2 = compute_losses(batch, out2, cfg, physics_scale=1.0)
        for k in keys:
            delta = float(losses2[k].detach().cpu()) - base[k]
            if abs(delta) < 1e-12:
                print(f'WARNING: {k} perturb_delta=0')
            print({'loss':k,'perturb_delta':delta})

if __name__ == '__main__':
    main()
