from __future__ import annotations

import json
import os
import os.path as osp
import random
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(osp.dirname(path))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def append_jsonl(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(osp.dirname(path))
    with open(path, 'a') as f:
        f.write(json.dumps(obj) + '\n')


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        return json.load(f)


def read_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def write_list(items: Iterable[str], path: str) -> None:
    ensure_dir(osp.dirname(path))
    with open(path, 'w') as f:
        for item in items:
            f.write(str(item) + '\n')


def npz_to_dict(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def patch_numpy_legacy_aliases() -> None:
    aliases = {
        'bool': bool,
        'int': int,
        'float': float,
        'complex': complex,
        'object': object,
        'unicode': str,
        'str': str,
    }
    for name, typ in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, typ)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def detach_to_float_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().cpu())
        elif isinstance(v, (int, float, np.number)):
            out[k] = float(v)
    return out


def mean_float_dict(dicts: List[Dict[str, Any]]) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    cnt: Dict[str, int] = {}
    for d in dicts:
        fd = detach_to_float_dict(d)
        for k, v in fd.items():
            acc[k] = acc.get(k, 0.0) + v
            cnt[k] = cnt.get(k, 0) + 1
    return {k: acc[k] / max(cnt[k], 1) for k in acc}


def aggregate_eval_metrics(dicts: List[Dict[str, Any]]) -> Dict[str, float]:
    out = mean_float_dict(dicts)
    if not dicts:
        return out

    def s(key: str) -> float:
        total = 0.0
        for d in dicts:
            if key in d:
                v = d[key]
                total += float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
        return total

    def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
        p = tp / max(tp + fp, 1.0)
        r = tp / max(tp + fn, 1.0)
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return p, r, f1

    tp, fp, fn = s('contact_tp'), s('contact_fp'), s('contact_fn')
    p, r, f1 = prf(tp, fp, fn)
    out['contact_tp'] = tp
    out['contact_fp'] = fp
    out['contact_fn'] = fn
    out['contact_precision'] = p
    out['contact_recall'] = r
    out['contact_f1'] = f1
    out['contact_valid_count'] = s('contact_valid_count')
    out['valid_contact_pair_count'] = s('valid_contact_pair_count')
    out['nan_window_count'] = s('nan_window_count')

    for th in ('0.05', '0.1', '0.2', '0.5', '0.8'):
        tpt, fpt, fnt = s(f'contact_tp@{th}'), s(f'contact_fp@{th}'), s(f'contact_fn@{th}')
        pt, rt, f1t = prf(tpt, fpt, fnt)
        out[f'contact_tp@{th}'] = tpt
        out[f'contact_fp@{th}'] = fpt
        out[f'contact_fn@{th}'] = fnt
        out[f'contact_precision@{th}'] = pt
        out[f'contact_recall@{th}'] = rt
        out[f'contact_f1@{th}'] = f1t
    return out
