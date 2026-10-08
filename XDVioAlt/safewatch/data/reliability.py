"""Modality-reliability signals for the adaptive fusion gate (P3/P8).

The subject asks for a fusion that adapts *selon la fiabilite des modalites*. Until now the gate was
trained with ``n_reliability=0``: it only saw the two embeddings and could never learn "ignore the
audio, it is saturated" or "ignore the video, the scene is black". This module supplies that missing
evidence, from data already on disk:

* **audio** - the per-patch statistics written next to every mel file by
  ``scripts/extract_audio_features.py`` (``data/features/audio_quality/<clip_id>.npy``, shape
  ``(P, 5)``: ``rms``, ``rms_db``, spectral ``flatness``, ``zcr``, ``clip_ratio``). Loudness,
  noise-likeness and saturation are exactly the degradations the subject lists.
* **visual** - *proxies* computed from the frozen feature stream, because the benchmark track never
  sees decoded frames: the RMS activation of the descriptor (a dark, blurry or occluded window
  produces a low-norm embedding) and the RMS of the snippet-to-snippet difference (a static or
  motion-blurred window barely changes). True brightness/blur/occlusion measurement needs pixels and
  belongs to the edge track (``docs/journal.md``).

Channels are standardised with *train-split* statistics (``scripts/quality_stats.py`` ->
``data/features/quality_stats.json``) so the gate sees comparable magnitudes and never test
statistics. A modality that is absent or deliberately degraded is encoded by the sentinel
:data:`DEGRADED` on every channel of that modality; the **same** substitution is used at training time
(modality dropout) and at scoring time (``--drop-modality``), so the gate meets at test time exactly
the pattern it was trained on - and the robustness study stays interpretable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

AUDIO_CHANNELS = ("rms", "rms_db", "flatness", "zcr", "clip_ratio")
AUDIO_DIM = len(AUDIO_CHANNELS)                                     # 5
VISUAL_CHANNELS = ("feature_energy", "temporal_motion")
VISUAL_DIM = len(VISUAL_CHANNELS)                                   # 2
RELIABILITY_DIM = AUDIO_DIM + VISUAL_DIM                            # 7

#: Sentinel written on every channel of a modality that is degraded/absent (standardised scale).
DEGRADED = -5.0


def audio_slice() -> slice:
    return slice(0, AUDIO_DIM)


def visual_slice() -> slice:
    return slice(AUDIO_DIM, RELIABILITY_DIM)


# --------------------------------------------------------------------------- #
# channels
# --------------------------------------------------------------------------- #
def visual_proxies(features: np.ndarray | None, n_snippets: int) -> np.ndarray:
    """``(T, D)`` frozen visual features -> ``(T, 2)`` reliability proxies (see module docstring)."""
    if features is None:
        return np.zeros((n_snippets, VISUAL_DIM), dtype=np.float32)
    f = np.asarray(features, dtype=np.float32)
    dim = max(1, f.shape[1])
    energy = np.linalg.norm(f, axis=1) / np.sqrt(dim)
    motion = np.zeros_like(energy)
    if f.shape[0] > 1:
        motion[1:] = np.linalg.norm(np.diff(f, axis=0), axis=1) / np.sqrt(dim)
    return np.stack([energy, motion], axis=1).astype(np.float32)


def pool_quality(quality: np.ndarray, n_snippets: int,
                 patches_per_snippet: float | None = None) -> np.ndarray:
    """Per-patch audio quality ``(P, 5)`` -> per-snippet ``(T, 5)`` means on the snippet grid.

    Uses the same pooling rule as the audio features themselves, so the gate sees the reliability of
    the very window it is scoring. ``patches_per_snippet`` must match the visual grid (mirror 2.78,
    official 0.69); ``None`` keeps the mirror default for backward compatibility.
    """
    from safewatch.data.multimodal import PATCHES_PER_SNIPPET, pool_audio

    pps = PATCHES_PER_SNIPPET if patches_per_snippet is None else patches_per_snippet
    return pool_audio(quality, n_snippets, pps)


def reliability_channels(quality: np.ndarray | None, visual: np.ndarray | None,
                         n_snippets: int, stats: dict | None = None,
                         patches_per_snippet: float | None = None) -> np.ndarray:
    """``(T, 7)`` = [audio quality (5) | visual proxies (2)], standardised when ``stats`` is given."""
    if quality is None or quality.size == 0:
        audio = np.zeros((n_snippets, AUDIO_DIM), dtype=np.float32)
    else:
        audio = pool_quality(quality, n_snippets, patches_per_snippet)
    out = np.concatenate([audio, visual_proxies(visual, n_snippets)], axis=1).astype(np.float32)
    if stats is not None:
        out = standardise(out, stats)
    return out


def standardise(values: np.ndarray, stats: dict,
                flat_channel_std: float = 1e-3) -> np.ndarray:
    """Z-score with the train-split mean/std stored by ``scripts/quality_stats.py``.

    A channel with (near-)zero variance in the train split carries no information the gate could
    learn from, and z-scoring it would amplify any test-time deviation by 1/std (``clip_ratio`` is
    exactly that on the mirror: no clipping at all in training). Such channels are passed through
    unchanged instead.
    """
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape[0] != values.shape[1]:
        raise ValueError(f"quality stats have {mean.shape[0]} channels, data has {values.shape[1]}")
    varying = std > flat_channel_std
    scaled = (values - mean) / np.maximum(std, 1e-6)
    return np.where(varying, scaled, values).astype(np.float32)


def load_quality_stats(path: str | Path | None) -> dict | None:
    """Read ``data/features/quality_stats.json``; ``None`` when absent (raw, unstandardised scale)."""
    if not path:
        return None
    stats_path = Path(path)
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# degradation (shared by training dropout and robustness scoring)
# --------------------------------------------------------------------------- #
def degrade_reliability(reliability: torch.Tensor | None, modality: str,
                        value: float = DEGRADED) -> torch.Tensor | None:
    """Mark one modality as degraded in a ``(B, T, 7)`` reliability tensor (no-op when ``None``)."""
    if reliability is None:
        return None
    if modality not in ("audio", "visual"):
        raise ValueError(f"unknown modality {modality!r}")
    out = reliability.clone()
    out[..., audio_slice() if modality == "audio" else visual_slice()] = value
    return out


def drop_modality(visual: torch.Tensor, audio: torch.Tensor, reliability: torch.Tensor | None,
                  modality: str | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Zero one modality's features **and** flag it in the reliability vector.

    The two halves must always travel together: a zeroed feature stream alone is out-of-distribution
    for the encoder, whereas the sentinel tells the gate *why* it is zeroed.
    """
    if modality in (None, "none"):
        return visual, audio, reliability
    if modality == "audio":
        return visual, torch.zeros_like(audio), degrade_reliability(reliability, "audio")
    if modality == "visual":
        return torch.zeros_like(visual), audio, degrade_reliability(reliability, "visual")
    raise ValueError(f"unknown modality to drop: {modality!r}")
