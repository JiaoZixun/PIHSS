from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch


def summarize_tensor(name: str, x: torch.Tensor) -> Dict[str, Any]:
    t = x.detach()
    return {
        'name': name,
        'shape': list(t.shape),
        'min': float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).min().item()) if t.numel() else 0.0,
        'max': float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).max().item()) if t.numel() else 0.0,
        'mean': float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).mean().item()) if t.numel() else 0.0,
        'std': float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).std(unbiased=False).item()) if t.numel() else 0.0,
        'nan_count': int(torch.isnan(t).sum().item()),
        'inf_count': int(torch.isinf(t).sum().item()),
    }


def _dump(dump_context: Dict[str, Any] | None, payload: Dict[str, Any]) -> None:
    if not dump_context:
        return
    out_dir = dump_context.get('dump_dir')
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    fn = f"nan_e{dump_context.get('epoch','x')}_b{dump_context.get('batch_idx','x')}.json"
    with open(os.path.join(out_dir, fn), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def assert_finite_tensor(name: str, x: torch.Tensor, dump_context: Dict[str, Any] | None = None) -> None:
    if not torch.isfinite(x).all():
        payload = {'failed': name, 'summary': summarize_tensor(name, x), **(dump_context or {})}
        _dump(dump_context, payload)
        raise FloatingPointError(f'Non-finite tensor: {name}')


def assert_finite_dict(prefix: str, d: Dict[str, Any], dump_context: Dict[str, Any] | None = None) -> None:
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            assert_finite_tensor(f'{prefix}.{k}', v, dump_context)


def check_model_params_finite(model: torch.nn.Module, dump_context: Dict[str, Any] | None = None) -> None:
    for n, p in model.named_parameters():
        assert_finite_tensor(f'param.{n}', p.data, dump_context)


def check_model_grads_finite(model: torch.nn.Module, dump_context: Dict[str, Any] | None = None) -> float:
    total = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        assert_finite_tensor(f'grad.{n}', p.grad, dump_context)
        total += float(p.grad.detach().norm().item())
    return total
