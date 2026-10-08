#!/usr/bin/env python
"""Compute per-channel mean/std of the reliability vector over the TRAIN split (adaptive fusion).

Writes ``data/features/quality_stats.json``, consumed by
``MultimodalClipDataset(quality_stats=...)`` -> :func:`safewatch.data.reliability.standardise`.

Why train-only: the same rule as ``scripts/audio_stats.py``. Standardising the gate's input with
test statistics would leak the test distribution into a quantity the model is trained on.

Channels (``safewatch/data/reliability.py``):
    0 rms        1 rms_db     2 flatness   3 zcr          4 clip_ratio    (from audio_quality)
    5 feature_energy             6 temporal_motion                      (visual proxies)

Usage:
    .venv/bin/python scripts/quality_stats.py
    .venv/bin/python scripts/quality_stats.py --limit 200      # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# `python scripts/quality_stats.py` puts scripts/ on sys.path, not the repo root: without this the
# `safewatch` package is not importable (the same defect made scripts/audio_stats.py unrunnable).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.data.reliability import (  # noqa: E402
    AUDIO_CHANNELS,
    RELIABILITY_DIM,
    VISUAL_CHANNELS,
    reliability_channels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/lists/train.csv")
    parser.add_argument("--quality-dir", default="data/features/audio_quality")
    parser.add_argument("--out", default="data/features/quality_stats.json")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N clips (debug)")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.train_csv, newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    quality_dir = Path(args.quality_dir)

    total = np.zeros(RELIABILITY_DIM, dtype=np.float64)
    total_sq = np.zeros(RELIABILITY_DIM, dtype=np.float64)
    count = 0
    used = 0
    missing_quality = 0
    for row in rows:
        quality_path = quality_dir / f"{row['clip_id']}.npy"
        quality = None
        if quality_path.exists():
            quality = np.asarray(np.load(quality_path), dtype=np.float32)
        else:
            missing_quality += 1
        visual = None
        visual_path = row.get("path") or row.get("prefix")
        if visual_path:
            if visual_path.endswith(".npy"):
                feats = np.load(visual_path, mmap_mode="r")
            else:  # official layout: <prefix>__0.npy .. <prefix>__4.npy -> (T, 5, D)
                feats = np.stack([np.asarray(np.load(f"{visual_path}__{c}.npy", mmap_mode="r"),
                                             dtype=np.float32) for c in range(5)], axis=1)
            if feats.ndim == 3:
                feats = np.asarray(feats, dtype=np.float32).mean(axis=1)
            visual = np.asarray(feats, dtype=np.float32)
        # raw scale here; standardisation happens later in the dataset, using these statistics
        channels = reliability_channels(quality, visual, int(row["T"]))
        total += channels.sum(axis=0, dtype=np.float64)
        total_sq += (channels.astype(np.float64) ** 2).sum(axis=0)
        count += channels.shape[0]
        used += 1

    if not used:
        print("[!] no clips processed - check the train list / feature paths")
        return 1
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
    names = list(AUDIO_CHANNELS) + list(VISUAL_CHANNELS)
    Path(args.out).write_text(json.dumps({
        "source": str(quality_dir),
        "train_csv": args.train_csv,
        "clips": used,
        "snippets": int(count),
        "missing_quality_files": missing_quality,
        "channels": names,
        "mean": np.round(mean, 6).tolist(),
        "std": np.round(std, 6).tolist(),
    }, indent=2), encoding="utf-8")
    print(f"[ok] {used} clips / {count} snippets -> {args.out} "
          f"(clips without quality stats: {missing_quality})")
    for name, mu, sigma in zip(names, mean, std, strict=True):
        print(f"     {name:16s} mean={mu:10.4f} std={sigma:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
