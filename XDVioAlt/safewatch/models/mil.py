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


class TemporalEncoder(nn.Module):
    """Conv1d stack over time: channel mixing (1x1) + local temporal context (kernel 5)."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128,
                 dropout: float = 0.6) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)  # (B,T,out_dim)


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
                 n_categories: int = 6, dropout: float = 0.6) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.encoder = TemporalEncoder(in_dim, hidden, emb, dropout)
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
