#!/usr/bin/env python
from __future__ import annotations
import argparse, os, os.path as osp
import pandas as pd
import torch
import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', 'src')))
from pinn_hoi.data.arctic_io import ArcticPICATSWindowDataset
from pinn_hoi.utils.io import ensure_dir, load_config

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    return p.parse_args()

def scan(ds, split):
    rows=[]
    for i in range(len(ds)):
        s=ds[i]
        gt=s['contact_label'][:-1]
        valid=torch.ones_like(gt)
        vc=int(valid.sum().item())
        pc=int((gt*valid).sum().item())
        ratio=pc/max(vc,1)
        rows.append({
            'split':split,'sample_idx':i,'file_path':s['seq_path'],'window_start':int(s['window_start'].item()),
            'contact_valid_count':vc,'gt_contact_pos_count':pc,'gt_contact_ratio':ratio,
        })
    return rows

def main():
    args=parse_args(); cfg=load_config(args.config)
    out=ensure_dir('outputs/debug_contact')
    train=ArcticPICATSWindowDataset(cfg['train_list'], int(cfg['window']), int(cfg['stride']), contact_thresh_m=float(cfg.get('contact_thresh_m',0.015)), preload=False)
    val=ArcticPICATSWindowDataset(cfg['val_list'], int(cfg['window']), int(cfg['window']), contact_thresh_m=float(cfg.get('contact_thresh_m',0.015)), preload=False)
    rows=scan(train,'train')+scan(val,'val')
    df=pd.DataFrame(rows).sort_values('gt_contact_ratio', ascending=False)
    df.to_csv(osp.join(out,'contact_sample_stats.csv'), index=False)
    top=df[df.gt_contact_pos_count>0].head(200)
    with open(osp.join(out,'top_positive_contact_samples.txt'),'w') as f:
        for _,r in top.iterrows():
            f.write(f"{r['split']} idx={int(r['sample_idx'])} ratio={r['gt_contact_ratio']:.6f} pos={int(r['gt_contact_pos_count'])} valid={int(r['contact_valid_count'])} {r['file_path']} st={int(r['window_start'])}\n")
    zero=df[df.gt_contact_pos_count==0]
    with open(osp.join(out,'no_contact_samples.txt'),'w') as f:
        for _,r in zero.iterrows():
            f.write(f"{r['split']} idx={int(r['sample_idx'])} {r['file_path']} st={int(r['window_start'])}\n")
    print(df.head(20)[['split','sample_idx','gt_contact_ratio','gt_contact_pos_count','contact_valid_count','file_path','window_start']].to_string(index=False))

if __name__=='__main__':
    main()
