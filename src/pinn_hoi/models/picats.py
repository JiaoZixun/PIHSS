from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pinn_hoi.common.rot import compose_axis_angle_delta


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class CrossAttentionBlock(nn.Module):
    """Bidirectional contact graph block between endpoint tokens and object point tokens."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.h_norm = nn.LayerNorm(dim)
        self.o_norm = nn.LayerNorm(dim)
        self.hand_to_obj = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.obj_to_hand = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.h_ffn = ResidualMLPBlock(dim, dropout=dropout)
        self.o_ffn = ResidualMLPBlock(dim, dropout=dropout)

    def forward(self, hand: torch.Tensor, obj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hand: [BT,E,D], obj: [BT,N,D]
        hq = self.h_norm(hand)
        okv = self.o_norm(obj)
        h_msg, attn_h2o = self.hand_to_obj(hq, okv, okv, need_weights=True, average_attn_weights=True)
        hand = hand + h_msg
        oq = self.o_norm(obj)
        hkv = self.h_norm(hand)
        o_msg, _ = self.obj_to_hand(oq, hkv, hkv, need_weights=False)
        obj = obj + o_msg
        hand = self.h_ffn(hand)
        obj = self.o_ffn(obj)
        # attn_h2o: [BT,E,N]
        return hand, obj, attn_h2o


class TemporalTransformer(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * ffn_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


class ContactAwareObjectMotionTransformer(nn.Module):
    """Top-conference style hand-object dynamics model.

    Inputs are preprocessed by UnifiedGeometryEngine. The model does not decode MANO itself during training; it consumes unified endpoints and object point clouds to avoid chain divergence between train/eval/vis.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        endpoint_token_dim: int = 128,
        object_token_dim: int = 128,
        num_object_tokens: int = 384,
        num_contact_blocks: int = 3,
        temporal_layers: int = 5,
        temporal_heads: int = 8,
        temporal_ffn_mult: int = 4,
        dropout: float = 0.1,
        use_contact_prior: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_object_tokens = num_object_tokens
        self.use_contact_prior = use_contact_prior

        # Endpoint token input per transition: endpoint xyz, velocity, hand id one-hot, joint id embedding.
        self.hand_id_embed = nn.Embedding(2, endpoint_token_dim)
        self.joint_id_embed = nn.Embedding(21, endpoint_token_dim)
        self.endpoint_in = MLP(6, hidden_dim, endpoint_token_dim, depth=3, dropout=dropout)
        self.endpoint_proj = nn.Linear(endpoint_token_dim, hidden_dim)

        # Object token input: current world xyz, canonical xyz, object center-relative xyz, approximated object point velocity.
        self.object_in = MLP(12, hidden_dim, object_token_dim, depth=3, dropout=dropout)
        self.object_proj = nn.Linear(object_token_dim, hidden_dim)

        self.pose_in = MLP(14, hidden_dim, hidden_dim, depth=3, dropout=dropout)
        self.contact_prior_proj = nn.Linear(1, hidden_dim)

        self.contact_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, heads=temporal_heads, dropout=dropout)
            for _ in range(num_contact_blocks)
        ])

        self.frame_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal = TemporalTransformer(hidden_dim, temporal_layers, temporal_heads, temporal_ffn_mult, dropout)

        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),
        )
        self.contact_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.impulse_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.friction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    @staticmethod
    def soft_contact_from_dist(min_dist: torch.Tensor, sigma: float = 0.015) -> torch.Tensor:
        return torch.exp(-(min_dist.clamp_min(0.0) ** 2) / (2.0 * sigma * sigma))

    def _select_object_tokens(self, obj_world: torch.Tensor, obj_canon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a fixed subset of object tokens for model efficiency.

        obj_world: [B,T,N,3], obj_canon: [B,N,3]
        """
        B, T, N, _ = obj_world.shape
        M = min(self.num_object_tokens, N)
        if M == N:
            idx = torch.arange(N, device=obj_world.device)
        else:
            # Preprocessed canonical points were already FPS-sampled; evenly spaced indices keep this deterministic and cheap.
            idx = torch.linspace(0, N - 1, M, device=obj_world.device).long()
        return obj_world[:, :, idx], obj_canon[:, idx], idx

    def _make_endpoint_tokens(self, hand_ep: torch.Tensor) -> torch.Tensor:
        # hand_ep: [B,T,2,21,3], return [B,T,42,D]
        B, T = hand_ep.shape[:2]
        cur = hand_ep[:, :-1]
        nxt = hand_ep[:, 1:]
        vel = nxt - cur
        x = torch.cat([cur, vel], dim=-1).reshape(B, T - 1, 42, 6)
        tok = self.endpoint_in(x)
        hand_ids = torch.arange(2, device=hand_ep.device).view(1, 1, 2, 1).expand(B, T - 1, 2, 21).reshape(B, T - 1, 42)
        joint_ids = torch.arange(21, device=hand_ep.device).view(1, 1, 1, 21).expand(B, T - 1, 2, 21).reshape(B, T - 1, 42)
        tok = tok + self.hand_id_embed(hand_ids) + self.joint_id_embed(joint_ids)
        return self.endpoint_proj(tok)

    def _make_object_tokens(self, obj_world: torch.Tensor, obj_canon: torch.Tensor, obj_pose: torch.Tensor) -> torch.Tensor:
        # obj_world: [B,T,N,3], obj_canon: [B,N,3], obj_pose: [B,T,7]
        B, T, N, _ = obj_world.shape
        ow_cur = obj_world[:, :-1]
        ow_nxt = obj_world[:, 1:]
        ov = ow_nxt - ow_cur
        center = obj_pose[:, :-1, 4:7].unsqueeze(2)
        rel = ow_cur - center
        canon = obj_canon.unsqueeze(1).expand(B, T - 1, N, 3)
        x = torch.cat([ow_cur, canon, rel, ov], dim=-1)
        return self.object_proj(self.object_in(x))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hand_ep = batch['hand_endpoints']          # [B,W,2,21,3]
        obj_pose = batch['obj_pose7']              # [B,W,7]
        obj_world_full = batch['obj_points_world'] # [B,W,N,3]
        obj_canon_full = batch['obj_points_canonical'] # [B,N,3]
        min_dist = batch['endpoint_obj_min_dist']  # [B,W,2,21]

        B, W = hand_ep.shape[:2]
        obj_world, obj_canon, obj_idx = self._select_object_tokens(obj_world_full, obj_canon_full)
        Htok = self._make_endpoint_tokens(hand_ep)                # [B,W-1,42,D]
        Otok = self._make_object_tokens(obj_world, obj_canon, obj_pose) # [B,W-1,M,D]

        if self.use_contact_prior:
            cp = self.soft_contact_from_dist(min_dist[:, :-1]).reshape(B, W - 1, 42, 1)
            Htok = Htok + self.contact_prior_proj(cp)

        # Fuse per-frame hand/object contact graph. Flatten B*(W-1).
        BT = B * (W - 1)
        Hf = Htok.reshape(BT, 42, self.hidden_dim)
        Of = Otok.reshape(BT, Otok.shape[2], self.hidden_dim)
        last_attn = None
        for blk in self.contact_blocks:
            Hf, Of, last_attn = blk(Hf, Of)
        Htok = Hf.reshape(B, W - 1, 42, self.hidden_dim)
        Otok = Of.reshape(B, W - 1, Otok.shape[2], self.hidden_dim)

        hand_mean = Htok.mean(dim=2)
        hand_max = Htok.max(dim=2).values
        obj_mean = Otok.mean(dim=2)
        pose_feat = self.pose_in(torch.cat([obj_pose[:, :-1], obj_pose[:, 1:] - obj_pose[:, :-1]], dim=-1))
        frame_token = self.frame_fuse(torch.cat([hand_mean, hand_max, obj_mean, pose_feat], dim=-1))
        temporal = self.temporal(frame_token)

        raw_delta = self.delta_head(temporal)
        # Keep deltas numerically conservative for stable PINN convergence. Translation is in meters.
        delta = torch.cat([
            0.05 * raw_delta[..., :1],
            0.15 * torch.tanh(raw_delta[..., 1:4]),
            0.10 * torch.tanh(raw_delta[..., 4:7]),
        ], dim=-1)
        # Do NOT update slices of pred_next in-place. The rotation branch needs
        # the base axis-angle tensor for autograd, and slice assignment changes
        # the version counter of the underlying tensor, causing backward errors.
        base_pose = obj_pose[:, :-1]
        pred_arti = base_pose[..., 0:1] + delta[..., 0:1]
        pred_rot = compose_axis_angle_delta(base_pose[..., 1:4], delta[..., 1:4])
        pred_trans = base_pose[..., 4:7] + delta[..., 4:7]
        pred_next = torch.cat([pred_arti, pred_rot, pred_trans], dim=-1)

        # Endpoint-conditioned outputs use temporal context added back to endpoint tokens.
        endpoint_ctx = Htok + temporal.unsqueeze(2)
        contact_logits = self.contact_head(endpoint_ctx).squeeze(-1).reshape(B, W - 1, 2, 21)
        impulse = self.impulse_head(endpoint_ctx).reshape(B, W - 1, 2, 21, 3)
        # mu in roughly [0.05, 1.55]
        mu = 0.05 + 1.5 * torch.sigmoid(self.friction_head(temporal)).squeeze(-1)
        uncertainty = self.uncertainty_head(temporal)

        return {
            'pred_next_pose7': pred_next,
            'delta_pose7': delta,
            'contact_logits': contact_logits,
            'contact_prob': torch.sigmoid(contact_logits),
            'impulse': impulse,
            'mu': mu,
            'uncertainty': uncertainty,
            'object_token_idx': obj_idx,
            'last_contact_attn': last_attn.reshape(B, W - 1, 42, -1) if last_attn is not None else None,
        }


def build_model_from_config(cfg: Dict) -> ContactAwareObjectMotionTransformer:
    mcfg = cfg.get('model', cfg)
    return ContactAwareObjectMotionTransformer(
        hidden_dim=int(mcfg.get('hidden_dim', 384)),
        endpoint_token_dim=int(mcfg.get('endpoint_token_dim', 128)),
        object_token_dim=int(mcfg.get('object_token_dim', 128)),
        num_object_tokens=int(mcfg.get('num_object_tokens', 384)),
        num_contact_blocks=int(mcfg.get('num_contact_blocks', 3)),
        temporal_layers=int(mcfg.get('temporal_layers', 5)),
        temporal_heads=int(mcfg.get('temporal_heads', 8)),
        temporal_ffn_mult=int(mcfg.get('temporal_ffn_mult', 4)),
        dropout=float(mcfg.get('dropout', 0.1)),
        use_contact_prior=bool(mcfg.get('use_contact_prior', True)),
    )
