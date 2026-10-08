#!/usr/bin/env python
"""Concatenate the two *complementary* visual representations into one feature file per clip.

Why: step 1 showed frozen CLIP is NOT a drop-in replacement for VideoSwin (0.7688 vs 0.7981), yet
the per-class breakdown showed they fail *differently* - CLIP is better on `shooting` (+0.031) and
`abuse` (+0.019), while VideoSwin is far better on the motion classes (`fighting` -0.181,
`explosion` -0.096).
Score-level averaging cannot exploit that (adding CLIP as an ensemble member *hurt*: 0.8191 vs
0.8216) because it forces a fixed 50/50 weight. Concatenating gives the head both views and lets it
learn the weighting.

Output is ``<out>/<split>/<clip_id>.npy`` of shape ``(T, 768 + 512)`` = ``(T, 1280)``, plus a split
CSV whose ``path`` points at it - the usual contract, so ``MultimodalClipDataset`` and
``safewatch.train_fusion`` consume it with ``--visual-dim 1280`` and no code change.

Usage:
    .venv/bin/python scripts/make_concat_features.py --splits train val test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_visual(path: str) -> np.ndarray:
    """The dataset's own crop rule: a ``(T, 5, D)`` official file becomes ``(T, D)`` by mean."""
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
    else:
        crops = [np.asarray(np.load(f"{path}__{c}.npy", mmap_mode="r"), dtype=np.float32)
                 for c in range(5)]
        arr = np.stack(crops, axis=1)
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    return np.asarray(arr, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lists", default="data/lists",
                        help="base lists; their `path` column points at the VideoSwin features")
    parser.add_argument("--clip-dir", default="data/features/clip",
                        help="CLIP features from scripts/extract_clip.py (same T grid)")
    parser.add_argument("--out", default="data/features/concat")
    parser.add_argument("--out-lists", default="data/lists_concat")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--clip-dim", type=int, default=512)
    args = parser.parse_args()

    list_root = Path(args.out_lists)
    list_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total = 0
    for split in args.splits:
        rows = list(csv.DictReader(open(Path(args.lists) / f"{split}.csv")))
        feat_dir = Path(args.out) / split
        feat_dir.mkdir(parents=True, exist_ok=True)
        out_rows, written, skipped, missing = [], 0, 0, 0
        for row in rows:
            clip_path = Path(args.clip_dir) / split / f"{row['clip_id']}.npy"
            target = feat_dir / f"{row['clip_id']}.npy"
            if not clip_path.exists():
                missing += 1
                continue
            clip = np.load(clip_path)
            if clip.shape[1] != args.clip_dim or clip.shape[0] != int(row["T"]):
                missing += 1
                continue
            out_rows.append({**row, "path": str(target)})
            if target.exists():
                skipped += 1
                continue
            swin = load_visual(row["path"])
            if swin.shape[0] != clip.shape[0]:
                print(f"[concat] SKIP {row['clip_id'][:44]}: swin T={swin.shape[0]} "
                      f"!= clip T={clip.shape[0]}", flush=True)
                out_rows.pop()
                continue
            feat = np.concatenate([swin, clip], axis=1).astype(np.float32)
            np.save(target, feat)
            written += 1
            if written % 250 == 0:
                print(f"[concat] {split}: {written} written "
                      f"[{time.time() - started:.0f}s]", flush=True)
        if out_rows:
            with (list_root / f"{split}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
                w.writeheader()
                w.writerows(out_rows)
        print(f"[concat] {split}: wrote {written} ({skipped} cached, {missing} missing) "
              f"-> {feat_dir} [{time.time() - started:.0f}s]", flush=True)
        total += written
    print(f"[concat] done: {total} features, lists -> {list_root} "
          f"in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
