"""Fusion heads for audio-visual anomaly detection (P3).

One model, four strategies, selected with ``fusion_mode`` so the ablation matrix is one flag:

* ``early``    - concatenate visual+audio snippet features, then a single encoder;
* ``late``     - separate encoders, snippet/clip scores averaged at the end;
* ``cross``    - bidirectional cross-attention between the two modalities;
* ``adaptive`` - cross-attention **plus** a per-snippet reliability gate ``alpha(t)``.

All variants return the same dictionary as :class:`~safewatch.models.mil.MILModel`, so training and
scoring code stay unchanged. ``alpha`` and the per-modality scores are also returned: they are the
"contribution of each modality" evidence the alert must display.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from safewatch.models.mil import AttentionPool, TemporalEncoder

FUSION_MODES = ("early", "late", "cross", "adaptive")


class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention: video queries audio, audio queries video.

    Supports SNR/reliability modulation: attenuates noisy/occluded modalities gracefully.
    """

    def __init__(self, dim: int = 128, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.v_from_a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.a_from_v = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                reliability: torch.Tensor | None = None):
        audio_in = audio
        if reliability is not None and reliability.shape[-1] >= 1:
            rel_scale = torch.sigmoid(reliability[:, :, :1]) if reliability.ndim == 3 else 1.0
            audio_in = audio * rel_scale
        v2, w_va = self.v_from_a(visual, audio_in, audio_in, key_padding_mask=pad_mask,
                                 need_weights=True)
        a2, w_av = self.a_from_v(audio, visual, visual, key_padding_mask=pad_mask,
                                 need_weights=True)
        return self.norm_v(visual + v2), self.norm_a(audio + a2), w_va, w_av


class AdaptiveGate(nn.Module):
    """Per-snippet mixing weight ``alpha(t)`` in [0, 1]: 1 = trust video, 0 = trust audio.

    Two input sources (``source``):

    * ``"embeddings"`` (default, the original design) - the two modality embeddings (agreement cue)
      plus the reliability features. This is the variant measured in Sessions 9-13: correct in
      spirit but the most seed-unstable of the four fusion modes (clean-model AP spans 0.096,
      six times its claimed occlusion benefit), which is why it must not be presented as a gain.
    * ``"reliability"`` - the reliability channels **only**. The gate then cannot memorise
      clip-specific embedding patterns: it is a pure measured-signal trust rule (degraded-video
      sentinel -> trust audio, degraded-audio sentinel -> trust video, otherwise let the signal
      statistics decide), with no data-dependent weights beyond a small MLP. That is the
      hypothesis behind the instability: the 0.096 spread came from the gate co-adapting to each
      seed's embeddings rather than from the fusion idea itself.
    """

    def __init__(self, dim: int = 128, n_reliability: int = 0, hidden: int = 64,
                 source: str = "embeddings") -> None:
        super().__init__()
        if source not in ("embeddings", "reliability"):
            raise ValueError(f"unknown gate source {source!r}; expected 'embeddings' or "
                             "'reliability'")
        self.source = source
        self.n_reliability = n_reliability
        in_width = n_reliability if source == "reliability" else 2 * dim + n_reliability
        if source == "reliability" and n_reliability <= 0:
            raise ValueError("gate source 'reliability' requires n_reliability > 0")
        self.net = nn.Sequential(nn.Linear(in_width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, visual: torch.Tensor, audio: torch.Tensor,
                reliability: torch.Tensor | None) -> torch.Tensor:
        if self.source == "reliability":
            if reliability is None:
                raise ValueError("this gate is driven by the reliability channels")
            return torch.sigmoid(self.net(reliability))  # (B, T, 1)
        parts = [visual, audio]
        if self.n_reliability > 0:
            if reliability is None:
                raise ValueError(f"this gate expects {self.n_reliability} reliability channels")
            parts.append(reliability)
        # else: a gate trained without reliability (n_reliability=0) silently ignores the argument,
        # so an old checkpoint can be scored with the current dataset without a shape clash
        return torch.sigmoid(self.net(torch.cat(parts, dim=-1)))  # (B, T, 1)


class FusionModel(nn.Module):
    """Audio-visual MIL model with a selectable fusion strategy."""

    def __init__(self, visual_dim: int = 768, audio_dim: int = 132, emb: int = 128,
                 hidden: int = 512, n_categories: int = 6, dropout: float = 0.6,
                 fusion_mode: str = "cross", heads: int = 4, n_reliability: int = 0,
                 gate_source: str = "embeddings") -> None:
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion_mode {fusion_mode!r}; expected {FUSION_MODES}")
        self.mode = fusion_mode
        self.n_reliability = n_reliability   # gate input width; 0 = gate blind to reliability
        self.gate_source = gate_source

        if fusion_mode == "early":
            self.encoder = TemporalEncoder(visual_dim + audio_dim, hidden, emb, dropout)
        else:
            self.visual_encoder = TemporalEncoder(visual_dim, hidden, emb, dropout)
            self.audio_encoder = TemporalEncoder(audio_dim, hidden, emb, dropout)
            if fusion_mode in ("cross", "adaptive"):
                self.cross = CrossModalAttention(emb, heads, dropout=dropout)
                self.merge = nn.Linear(2 * emb, emb)
        if fusion_mode == "adaptive":
            self.gate = AdaptiveGate(emb, n_reliability, source=gate_source)

        self.pool = AttentionPool(emb)
        self.snippet_head = nn.Conv1d(emb, 1, kernel_size=1)
        self.clip_head = nn.Linear(emb, 1)
        self.multi_head = nn.Linear(emb, n_categories)
        if fusion_mode == "late":                       # one snippet head per modality
            self.visual_snippet_head = nn.Conv1d(emb, 1, kernel_size=1)
            self.audio_snippet_head = nn.Conv1d(emb, 1, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor,
                mask: torch.Tensor | None = None, reliability: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device)
        pad_mask = ~mask              # MultiheadAttention convention: True means "ignore"

        alpha = None
        snippet_visual = snippet_audio = None

        if self.mode == "early":
            fused = self.encoder(torch.cat([visual, audio], dim=-1))
        else:
            emb_v = self.visual_encoder(visual)
            emb_a = self.audio_encoder(audio)
            if self.mode == "late":
                fused = 0.5 * (emb_v + emb_a)
                snippet_visual = self.visual_snippet_head(emb_v.permute(0, 2, 1)).squeeze(1)
                snippet_audio = self.audio_snippet_head(emb_a.permute(0, 2, 1)).squeeze(1)
            else:
                emb_v, emb_a, _w_va, _w_av = self.cross(
                    emb_v, emb_a, pad_mask=pad_mask, reliability=reliability
                )
                if self.mode == "adaptive":
                    alpha = self.gate(emb_v, emb_a, reliability)
                    fused = self.merge(torch.cat([alpha * emb_v, (1.0 - alpha) * emb_a], dim=-1))
                else:
                    fused = self.merge(torch.cat([emb_v, emb_a], dim=-1))

        pooled, weights, attn_logits = self.pool(fused, mask)
        snippet_logits = self.snippet_head(fused.permute(0, 2, 1)).squeeze(1)
        if self.mode == "late":       # late fusion: combine the two unimodal score curves
            snippet_logits = 0.5 * (snippet_logits + 0.5 * (snippet_visual + snippet_audio))

        return {
            "snippet_logits": snippet_logits,
            "clip_logits": self.clip_head(pooled).squeeze(-1),
            "multi_logits": self.multi_head(pooled),
            "attn": weights,
            "attn_logits": attn_logits,
            "embeddings": fused,
            "alpha": alpha,
            "snippet_visual": snippet_visual,
            "snippet_audio": snippet_audio,
        }

    @staticmethod
    def snippet_scores(snippet_logits: torch.Tensor) -> torch.Tensor:
        """Per-snippet anomaly probability (the score curve shown in the operator UI)."""
        return torch.sigmoid(snippet_logits)
