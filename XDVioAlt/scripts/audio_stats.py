#!/usr/bin/env python
"""Compute per-dimension mean/std of the pooled audio features over the TRAIN split.

Written to ``data/features/audio_stats.json`` and consumed by
``MultimodalClipDataset(audio_stats=...)`` so that audio normalisation uses training statistics only
(never test statistics - that would leak).

Usage:
    .venv/bin/python scripts/audio_stats.py --audio-dir data/features/mel
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# `python scripts/audio_stats.py` puts scripts/ on sys.path, not the repo root, so the safewatch
# package was not importable with the documented invocation (fixed here and in quality_stats.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.data.multimodal import align_audio_to_snippets, pool_audio  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default="data/features/mel")
    parser.add_argument("--train-csv", default="data/lists/train.csv")
    parser.add_argument("--audio-grid", default="patch", choices=("patch", "snippet"),
                        help="'patch' = our 0.96 s mel hop, pooled onto the snippet grid; "
                             "'snippet' = already on it (official VGGish), aligned as-is")
    parser.add_argument("--out", default="data/features/audio_stats.json")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.train_csv, newline="", encoding="utf-8")))
    audio_dir = Path(args.audio_dir)
    total = None
    total_sq = None
    count = 0
    used = 0
    for row in rows:
        path = audio_dir / f"{row['clip_id']}.npy"
        if not path.exists():
            continue
        pooled = (align_audio_to_snippets(np.load(path).astype(np.float32), int(row["T"]))
                  if args.audio_grid == "snippet" else
                  pool_audio(np.load(path).astype(np.float32), int(row["T"])))  # (T, A)
        if total is None:
            total = np.zeros(pooled.shape[1], dtype=np.float64)
            total_sq = np.zeros(pooled.shape[1], dtype=np.float64)
        total += pooled.sum(axis=0)
        total_sq += (pooled.astype(np.float64) ** 2).sum(axis=0)
        count += pooled.shape[0]
        used += 1

    if not used:
        print(f"[!] no audio features found in {audio_dir} - run the extraction first")
        return 1
    mean = total / count
    var = np.maximum(total_sq / count - mean ** 2, 1e-12)
    std = np.sqrt(var)
    Path(args.out).write_text(json.dumps({
        "source": str(audio_dir),
        "clips": used,
        "snippets": int(count),
        "mean": np.round(mean, 6).tolist(),
        "std": np.round(std, 6).tolist(),
    }, indent=2), encoding="utf-8")
    print(f"[ok] {used} clips / {count} snippets -> {args.out}")
    print(f"[info] std range: {std.min():.3f} .. {std.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
