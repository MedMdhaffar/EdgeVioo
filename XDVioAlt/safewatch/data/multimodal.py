"""Multimodal (visual + audio) clip dataset for XD-Violence.

Audio features are stored per 0.96 s patch (VGGish hop) while the visual grid advances every 64
frames (2.667 s at 24 fps), so the patches must be pooled onto the snippet grid; getting this wrong
shifts audio relative to video by seconds and quietly destroys fusion (docs/dataset.md).

The class inherits from :class:`~safewatch.data.dataset.ClipDataset` but re-implements ``__getitem__``
so that **one** crop/pad decision is applied to every modality (the parent draws a fresh random crop
per call, which would misalign audio and video).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from safewatch.data.dataset import CATEGORIES, ClipDataset
from safewatch.data.reliability import load_quality_stats, reliability_channels

AUDIO_PATCH_SECONDS = 0.96
SNIPPET_FRAMES = 64.0
SNIPPET_FPS = 24.0
PATCHES_PER_SNIPPET = (SNIPPET_FRAMES / SNIPPET_FPS) / AUDIO_PATCH_SECONDS  # ~2.78
AUDIO_DIM = 132   # log-mel patch statistics - one source of truth (scripts/extract_audio_features.py)
SILENCE_RMS_DB = -75.0  # the extractor's exact digital-silence floor is -90 dB

DEFAULT_QUALITY_DIR = "data/features/audio_quality"
DEFAULT_QUALITY_STATS = "data/features/quality_stats.json"


def align_audio_to_snippets(audio: np.ndarray, n_snippets: int) -> np.ndarray:
    """``(T, A)`` features already on the *snippet* grid -> exactly ``n_snippets`` rows.

    The official XD-Violence VGGish release is extracted per snippet (its row count equals the I3D
    snippet count), so pooling it with :func:`pool_audio` - which assumes the 0.96 s patch hop of our
    own mel features - would compress it by ~2.8x and shift audio against video. Identity is the
    correct mapping; a ±1 row difference (rounding in the official extractor) is repaired by nearest
    neighbour resampling rather than silently truncated, and a 0-row file degrades to zeros.
    """
    if audio.ndim == 1:
        audio = audio[:, None]
    rows, dim = int(audio.shape[0]), int(audio.shape[1])
    if rows == 0:
        return np.zeros((n_snippets, dim), dtype=np.float32)
    if rows == n_snippets:
        return audio.astype(np.float32, copy=False)
    index = np.linspace(0, rows - 1, n_snippets).round().astype(int)
    return audio[index].astype(np.float32, copy=False)


def patches_per_snippet(stride_frames: float, fps: float = SNIPPET_FPS) -> float:
    """How many 0.96 s audio patches fit in one visual snippet of ``stride_frames`` frames.

    The mirror grid (64 frames = 2.67 s) packs ~2.78 patches; the official grid (16 frames =
    0.67 s) packs ~0.69. Using one constant for both silently slides the audio track off the video
    on whichever grid it was not calibrated for, so the grid must be stated, never assumed.
    """
    if stride_frames <= 0:
        raise ValueError("stride_frames must be positive")
    return (stride_frames / fps) / AUDIO_PATCH_SECONDS


def pool_audio(audio: np.ndarray, n_snippets: int,
               patches_per_snippet: float = PATCHES_PER_SNIPPET) -> np.ndarray:
    """``(P, A)`` patches at the 0.96 s hop -> ``(n_snippets, A)`` means on the snippet grid.

    ``patches_per_snippet`` must match the *visual* grid the audio is fused with (see
    :func:`patches_per_snippet`); the default is the mirror grid, kept for backward compatibility.
    """
    if audio.shape[0] == 0:
        return np.zeros((n_snippets, max(1, audio.shape[1] if audio.ndim == 2 else 1)),
                        dtype=np.float32)
    patch_index = np.arange(audio.shape[0])
    snippet_of_patch = np.minimum((patch_index / patches_per_snippet).astype(int), n_snippets - 1)
    dim = audio.shape[1]
    sums = np.zeros((n_snippets, dim), dtype=np.float64)
    counts = np.zeros(n_snippets, dtype=np.float64)
    np.add.at(sums, snippet_of_patch, audio)
    np.add.at(counts, snippet_of_patch, 1.0)
    return (sums / np.maximum(counts[:, None], 1.0)).astype(np.float32)


def audit_audio_rows(rows: list[dict], audio_dir: str | Path, audio_grid: str,
                     expected_dim: int, stride_frames: float,
                     fps: float = SNIPPET_FPS,
                     quality_dir: str | Path | None = None) -> dict:
    """Check audio files and signal health against a split before training or scoring starts."""
    if audio_grid not in ("patch", "snippet"):
        raise ValueError(f"unknown audio_grid {audio_grid!r}")
    if expected_dim < 1 or stride_frames <= 0 or fps <= 0:
        raise ValueError("expected_dim, stride_frames, and fps must be positive")

    audio_dir = Path(audio_dir)
    quality_dir = Path(quality_dir) if quality_dir else None
    counts = {"clips": len(rows), "present": 0, "missing": 0, "load_error": 0, "empty": 0,
              "wrong_ndim": 0, "wrong_dim": 0, "non_finite": 0, "misaligned": 0}
    counts.update({"quality_missing": 0, "quality_load_error": 0, "quality_invalid": 0,
                   "silent_audio": 0, "constant_features": 0})
    examples, issues, observed_rows, expected_row_counts = [], [], [], []

    def record(code: str, row: dict, detail: str) -> None:
        counts[code] += 1
        if len(issues) < 1000:
            issues.append({"clip_id": row["clip_id"], "split": row.get("split", ""),
                           "csv": row.get("_source_csv"), "issue": code, "detail": detail})
        if len(examples) < 8:
            examples.append(f"{row['clip_id']}: {code} ({detail})")

    for row in rows:
        path = audio_dir / f"{row['clip_id']}.npy"
        if not path.is_file():
            record("missing", row, str(path))
            continue
        counts["present"] += 1
        try:
            features = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            record("load_error", row, f"{type(exc).__name__}: {exc}")
            continue
        if features.ndim != 2:
            record("wrong_ndim", row, f"shape={features.shape}")
            continue
        n_rows, dim = features.shape
        if n_rows == 0:
            record("empty", row, "zero time rows")
        if dim != expected_dim:
            record("wrong_dim", row, f"expected {expected_dim}, got {dim}")
        if not np.isfinite(features).all():
            record("non_finite", row, "contains NaN or infinity")

        n_snippets = int(row["T"])
        if audio_grid == "snippet":
            expected_count = n_snippets
            tolerance = 1  # loader repairs a single-row rounding discrepancy
        else:
            patch_seconds = AUDIO_PATCH_SECONDS
            expected_count = n_snippets * (stride_frames / fps) / patch_seconds
            tolerance = int(np.ceil((stride_frames / fps) / patch_seconds)) + 1
        if abs(n_rows - expected_count) > tolerance:
            record("misaligned", row,
                   f"observed {n_rows} rows; expected about {expected_count:.2f} "
                   f"(tolerance ±{tolerance})")

        observed_rows.append(n_rows)
        expected_row_counts.append(expected_count)

        if n_rows >= 2 and np.all(np.std(features, axis=0) <= 1e-8):
            record("constant_features", row, "all feature channels are constant over time")

        if quality_dir is not None:
            quality_path = quality_dir / f"{row['clip_id']}.npy"
            if not quality_path.is_file():
                record("quality_missing", row, str(quality_path))
                continue
            try:
                quality = np.load(quality_path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                record("quality_load_error", row, f"{type(exc).__name__}: {exc}")
                continue
            if quality.ndim != 2 or quality.shape[0] < 1 or quality.shape[1] < 2:
                record("quality_invalid", row, f"expected nonempty (patches, >=2), got {quality.shape}")
                continue
            if not np.isfinite(quality).all():
                record("quality_invalid", row, "quality features contain NaN or infinity")
                continue
            if np.all(quality[:, 1] <= SILENCE_RMS_DB):
                record("silent_audio", row,
                       f"all patches at or below {SILENCE_RMS_DB:g} dB RMS")

    row_ids = {row["clip_id"] for row in rows}
    orphan_count = sum(1 for path in audio_dir.glob("*.npy") if path.stem not in row_ids)
    return {"audio_dir": str(audio_dir), "audio_grid": audio_grid,
            "expected_dim": expected_dim, "stride_frames": stride_frames,
            "quality_dir": str(quality_dir) if quality_dir is not None else None,
            "counts": counts, "examples": examples, "issues": issues,
            "issues_truncated": sum(v for key, v in counts.items()
                                     if key not in ("clips", "present")) > len(issues),
            "orphan_feature_files": orphan_count,
            "time_rows": ({"observed_min": int(min(observed_rows)),
                           "observed_median": float(np.median(observed_rows)),
                           "observed_max": int(max(observed_rows)),
                           "expected_median": float(np.median(expected_row_counts))}
                          if observed_rows else None)}


def report_audio_preflight(report: dict, label: str) -> None:
    """Print a concise PASS/WARN summary; dataset issues remain visible in run logs."""
    counts = report["counts"]
    failures = {key: value for key, value in counts.items()
                if key not in ("clips", "present") and value}
    if failures:
        summary = ", ".join(f"{key}={value}" for key, value in failures.items())
        print(f"[audio-audit] WARN {label}: {summary}")
        for example in report["examples"]:
            print(f"[audio-audit]   {example}")
    else:
        print(f"[audio-audit] PASS {label}: {counts['clips']} clips, "
              f"grid={report['audio_grid']}, dim={report['expected_dim']}, "
              f"stride={report['stride_frames']:g}")


def clip_normalise(pooled: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Instance-normalise each audio channel over the clip's time axis ``(T, A)``.

    This is the fix for the diagnosed calibration defect (``docs/journal.md`` Session 9): audio score
    *levels* are not comparable between clips (a loud film scores high everywhere, a quiet one
    everywhere low), so pooling snippets across clips destroys the global PR curve. Removing the
    per-clip gain and offset makes the audio stream speak the same language clip to clip, while
    keeping the within-clip ranking - which is exactly what the localisation needs.
    """
    mean = pooled.mean(axis=0, keepdims=True)
    std = pooled.std(axis=0, keepdims=True)
    return ((pooled - mean) / np.maximum(std, eps)).astype(np.float32)


def load_audio_stats(path: str | Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-dimension (mean, std) for audio normalisation, as written by ``scripts/audio_stats.py``."""
    if not path:
        return None
    stats_path = Path(path)
    if not stats_path.exists():
        return None
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return (np.asarray(stats["mean"], dtype=np.float32),
            np.asarray(stats["std"], dtype=np.float32))


class MultimodalClipDataset(ClipDataset):
    """Clip dataset returning visual features, audio features and one shared mask.

    ``audio_dir`` holds per-clip ``<clip_id>.npy`` files of shape ``(P, A)`` at the 0.96 s hop;
    ``modality`` selects what is actually loaded (``"visual"`` is fastest, ``"audio"`` trains a
    unimodal audio model, ``"both"`` is the fusion setting).

    ``audio_grid`` says which time grid those files live on: ``"patch"`` (default) is our own 0.96 s
    mel hop, which is pooled onto the snippet grid; ``"snippet"`` is the official XD-Violence VGGish
    release, which is *already* one row per snippet and must not be pooled (see
    :func:`align_audio_to_snippets`). Mixing these up is silent: shapes stay plausible while the
    audio track drifts seconds away from the video.

    With ``modality="both"`` every item also carries ``reliability`` ``(T, 7)``: the audio signal
    statistics pooled onto the snippet grid plus two visual degradation proxies, standardised with
    train-split statistics. That tensor is what makes the adaptive fusion gate adaptive
    (:mod:`safewatch.data.reliability`).

    ``audio_norm="clip"`` additionally z-scores each audio channel **within the clip**
    (instance normalisation over time). Diagnosed in ``docs/journal.md`` Session 9: the audio score
    curve ranks well inside a clip but its levels are not comparable between clips (global AP 0.446
    against 0.625 per-clip; removing audio even *raises* fused global AP), which is a gain/offset
    problem, not a capacity problem. Note the statistics are taken over the whole clip, which is fine
    for the offline benchmark and must become cumulative/streaming for online inference.
    """

    def __init__(self, csv_path, audio_dir, modality: str = "both",
                 audio_stats: str | Path | None = None,
                 quality_dir: str | Path | None = DEFAULT_QUALITY_DIR,
                 quality_stats: str | Path | None = DEFAULT_QUALITY_STATS,
                 audio_norm: str = "none", audio_grid: str = "patch",
                 stride_frames: float | None = None, **kwargs) -> None:
        super().__init__(csv_path, **kwargs)
        if modality not in ("visual", "audio", "both"):
            raise ValueError(f"unknown modality {modality!r}")
        if audio_norm not in ("none", "clip"):
            raise ValueError(f"unknown audio_norm {audio_norm!r}")
        if audio_grid not in ("patch", "snippet"):
            raise ValueError(f"unknown audio_grid {audio_grid!r}; expected 'patch' or 'snippet'")
        self.audio_dir = Path(audio_dir)
        self.modality = modality
        self.audio_stats = load_audio_stats(audio_stats)
        self.audio_norm = audio_norm
        self.audio_grid = audio_grid
        # Which visual grid the 0.96 s audio patches are pooled onto. ``None`` keeps the mirror
        # (64-frame) default; a 16-frame grid MUST pass 16 so the audio does not slide by 4x.
        self.patches_per_snippet = (PATCHES_PER_SNIPPET if not stride_frames
                                    else patches_per_snippet(stride_frames))
        self.quality_dir = Path(quality_dir) if quality_dir else None
        self.quality_stats = load_quality_stats(quality_stats)
        self._warned_missing = False
        self._warned_quality = False

    def _index_map(self, n_snippets: int) -> np.ndarray:
        """Output row -> source snippet index (-1 = padding); drawn ONCE per sample."""
        if self.max_snippets is None:
            return np.arange(n_snippets)
        n = self.max_snippets
        if n_snippets >= n:
            if self.train:  # random crop (training only): a different window each epoch
                start = int(self.rng.integers(0, n_snippets - n + 1))
                return np.arange(start, start + n)
            return np.linspace(0, n_snippets - 1, n).round().astype(int)
        return np.concatenate([np.arange(n_snippets), -np.ones(n - n_snippets, dtype=int)])

    @staticmethod
    def _gather(array: np.ndarray, index_map: np.ndarray) -> np.ndarray:
        """Apply a crop/pad index map to a ``(T, D)`` array (padding rows remain zero)."""
        out = np.zeros((index_map.shape[0], array.shape[1]), dtype=np.float32)
        valid = index_map >= 0
        out[valid] = array[index_map[valid]]
        return out

    def _audio(self, clip_id: str, n_snippets: int, index_map: np.ndarray) -> np.ndarray:
        path = self.audio_dir / f"{clip_id}.npy"
        dim = int(self.audio_stats[0].shape[0]) if self.audio_stats is not None else AUDIO_DIM
        if not path.exists():
            if not self._warned_missing:
                print(f"[warn] missing audio features for {clip_id[:40]} -> zeros "
                      f"(run scripts/extract_audio_features.py)")
                self._warned_missing = True
            pooled = np.zeros((n_snippets, dim), dtype=np.float32)
        else:
            features = np.load(path).astype(np.float32)
            pooled = (align_audio_to_snippets(features, n_snippets) if self.audio_grid == "snippet"
                      else pool_audio(features, n_snippets, self.patches_per_snippet))
        if self.audio_stats is not None and pooled.shape[1] == self.audio_stats[0].shape[0]:
            pooled = (pooled - self.audio_stats[0]) / np.maximum(self.audio_stats[1], 1e-6)
        if self.audio_norm == "clip":
            pooled = clip_normalise(pooled)
        return self._gather(pooled, index_map)

    def _reliability(self, clip_id: str, n_snippets: int,
                     visual: np.ndarray | None) -> np.ndarray:
        """``(T, RELIABILITY_DIM)`` on the source snippet grid (gathered by the caller)."""
        quality = None
        if self.quality_dir is not None:
            path = self.quality_dir / f"{clip_id}.npy"
            if path.exists():
                quality = np.asarray(np.load(path), dtype=np.float32)
            elif not self._warned_quality:
                print(f"[warn] missing audio quality stats for {clip_id[:40]} -> zeros "
                      f"(run scripts/extract_audio_features.py)")
                self._warned_quality = True
        return reliability_channels(quality, visual, n_snippets, self.quality_stats,
                                    self.patches_per_snippet)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        n_snippets = int(row["T"])
        index_map = self._index_map(n_snippets)
        mask = index_map >= 0

        multi = np.zeros(len(CATEGORIES), dtype=np.float32)
        for code in (row["labels"] or "").split("|"):
            if code in CATEGORIES:
                multi[CATEGORIES.index(code)] = 1.0

        item: dict = {
            "clip_id": row["clip_id"],
            "n_snippets": n_snippets,
            "mask": torch.from_numpy(mask),
            "binary": torch.tensor(float(row["binary"]), dtype=torch.float32),
            "multi": torch.from_numpy(multi),
        }
        visual = None
        if self.modality in ("visual", "both"):
            visual = self._load(row.get("path") or row.get("prefix"))
            item["visual"] = torch.from_numpy(self._gather(visual, index_map))
        if self.modality in ("audio", "both"):
            item["audio"] = torch.from_numpy(self._audio(row["clip_id"], n_snippets, index_map))
        if self.modality == "both":   # the adaptive gate's evidence, aligned with the same window
            item["reliability"] = torch.from_numpy(
                self._gather(self._reliability(row["clip_id"], n_snippets, visual), index_map))
        return item
