"""Visual-only MIL baseline - the P1 reference model.

Pipeline:  ``(B,T,D)`` -> temporal encoder -> snippet embeddings ``(B,T,C)``
           -> snippet head (per-moment logits: localisation + top-k MIL)
           -> masked temporal attention -> pooled clip logit -> 6 category logits.

Deliberately small (~0.5 M params, measured ~24 ms per training step on this CPU) so that an epoch
over the full train split takes seconds. The P3 cross-modal attention module will replace the
encoder's fusion stage while keeping this structure, so the baseline stays comparable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from safewatch.losses.mil import masked_softmax


class MultiScaleDilatedTCN(nn.Module):
    """Parallel dilated temporal convolutions at d∈{1,2,4}.

    Replaces the single k=5 Conv1d in :class:`TemporalEncoder` when ``use_ms_tcn=True``.
    Three branches share the input and are concatenated along the channel axis:

    * d=1 (42/44 ch) — sub-snippet micro-shocks and sudden impacts (< 0.67 s)
    * d=2 (42 ch)    — medium transitions and onset build-ups (~ 1.3 s)
    * d=4 (44 ch)    — extended pileups and multi-vehicle incidents (~ 2.7 s at k4_mstd)

    Output channel count equals ``dim`` regardless of the split, and the output is
    immediately activated with ReLU, so it drops into the existing Sequential in place of
    the old ``Conv1d + ReLU`` pair without changing downstream layer shapes.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        d1 = dim // 3          # e.g. 42 for dim=128
        d2 = dim // 3          # e.g. 42
        d4 = dim - d1 - d2     # e.g. 44  →  total always equals dim
        self.b1 = nn.Conv1d(dim, d1, kernel_size=3, padding=1,  dilation=1)
        self.b2 = nn.Conv1d(dim, d2, kernel_size=3, padding=2,  dilation=2)
        self.b4 = nn.Conv1d(dim, d4, kernel_size=3, padding=4,  dilation=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        return self.act(torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1))


class TemporalEncoder(nn.Module):
    """Conv1d stack over time: channel mixing (1x1) + local temporal context.

    When ``use_ms_tcn=False`` (default) the final stage is the original single-scale
    k=5 Conv1d — existing checkpoint weights load without modification.

    When ``use_ms_tcn=True`` the final stage is replaced by :class:`MultiScaleDilatedTCN`
    (parallel d∈{1,2,4} branches). This path uses different weight names
    (``front.*`` / ``ms_tcn.*`` instead of ``net.*``) and must be trained from scratch.
    """

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128,
                 dropout: float = 0.6, use_ms_tcn: bool = False) -> None:
        super().__init__()
        self.use_ms_tcn = use_ms_tcn
        if use_ms_tcn:
            # Separate attributes so checkpoint keys don't collide with the old ``net``
            self.front = nn.Sequential(
                nn.Conv1d(in_dim, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, out_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.ms_tcn = MultiScaleDilatedTCN(out_dim)
        else:
            # Original architecture — attribute name ``net`` preserved for checkpoint compat.
            self.net = nn.Sequential(
                nn.Conv1d(in_dim, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, out_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(out_dim, out_dim, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,in_dim) -> (B,T,out_dim)
        x_t = x.transpose(1, 2)                           # -> (B, in_dim, T)
        if self.use_ms_tcn:
            return self.ms_tcn(self.front(x_t)).transpose(1, 2)
        return self.net(x_t).transpose(1, 2)


class AttentionPool(nn.Module):
    """Masked temporal attention; the weights double as temporal evidence for the alerts."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        logits = self.score(x).squeeze(-1)  # (B,T)
        weights = masked_softmax(logits, mask, dim=1)
        pooled = torch.einsum("bt,btc->bc", weights, x)
        return pooled, weights, logits


class MILModel(nn.Module):
    """Weakly-supervised visual model: per-snippet scores + clip score + category logits."""

    def __init__(self, in_dim: int = 768, hidden: int = 512, emb: int = 128,
                 n_categories: int = 6, dropout: float = 0.6,
                 use_ms_tcn: bool = False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.encoder = TemporalEncoder(in_dim, hidden, emb, dropout, use_ms_tcn=use_ms_tcn)
        self.pool = AttentionPool(emb)
        self.snippet_head = nn.Conv1d(emb, 1, kernel_size=1)
        self.clip_head = nn.Linear(emb, 1)
        self.multi_head = nn.Linear(emb, n_categories)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        emb = self.encoder(features)
        pooled, weights, attn_logits = self.pool(emb, mask)
        snippet_logits = self.snippet_head(emb.permute(0, 2, 1)).squeeze(1)  # (B,T)
        return {
            "snippet_logits": snippet_logits,
            "clip_logits": self.clip_head(pooled).squeeze(-1),
            "multi_logits": self.multi_head(pooled),
            "attn": weights,
            "attn_logits": attn_logits,
            "embeddings": emb,
        }

    @staticmethod
    def snippet_scores(snippet_logits: torch.Tensor) -> torch.Tensor:
        """Per-snippet anomaly probability (the score curve shown in the UI)."""
        return torch.sigmoid(snippet_logits)
