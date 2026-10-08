#!/usr/bin/env python
"""Pool the official-grid VGGish snippet features onto the mirror grid, as a drop-in audio source.

Why: on the *official* grid, swapping log-mel 132-d for real VGGish 128-d turned audio from a
"passenger" into a load-bearing modality - the fusion margin jumped from +0.0025 to +0.0486 global
(README, P3 / P2 rows). The mirror grid, which still holds the project's best global AP (0.8125),
never got that swap: it feeds the weaker log-mel statistics. This script builds the pooled features
so `--audio-dir data/features/vggish_mirror --audio-dim 128 --audio-grid snippet` works unchanged.

The official grid advances one snippet every 16 frames (0.667 s); the mirror grid every 64 frames
(2.667 s), so exactly 4 official snippets fold into one mirror snippet. Pooling is a windowed mean
over those 4 rows - *not* nearest-neighbour resampling, which would throw away 3 of every 4.

Usage:
    .venv/bin/python scripts/pool_vggish_mirror.py --splits train val test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pool_to(features: np.ndarray, n_snippets: int) -> np.ndarray:
    """``(T_off, D)`` official rows -> ``(n_snippets, D)`` mirror rows, windowed mean."""
    rows = int(features.shape[0])
    if rows == n_snippets:
        return features.astype(np.float32)
    if rows < n_snippets:      # nothing to fold; nearest neighbour is the only option
        index = np.linspace(0, rows - 1, n_snippets).round().astype(int)
        return features[index].astype(np.float32)
    # edges[i] .. edges[i+1] is the official span covered by mirror snippet i
    edges = np.linspace(0, rows, n_snippets + 1).round().astype(int)
    out = np.empty((n_snippets, features.shape[1]), dtype=np.float32)
    for i in range(n_snippets):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out[i] = features[lo:hi].mean(axis=0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lists", default="data/lists",
                        help="mirror lists; column `T` is the mirror snippet count")
    parser.add_argument("--vggish", default="data/features/vggish_snippet",
                        help="official-grid VGGish features from the release")
    parser.add_argument("--out", default="data/features/vggish_mirror")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    started = time.time()
    total = 0
    for split in args.splits:
        rows = list(csv.DictReader(open(Path(args.lists) / f"{split}.csv")))
        out_dir = Path(args.out) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        written, skipped, missing = 0, 0, 0
        for row in rows:
            source = Path(args.vggish) / f"{row['clip_id']}.npy"
            target = out_dir / f"{row['clip_id']}.npy"
            if not source.exists():
                missing += 1
                continue
            if target.exists():
                skipped += 1
                continue
            pooled = pool_to(np.load(source), int(row["T"]))
            np.save(target, pooled)
            written += 1
            if written % 500 == 0:
                print(f"[vggish] {split}: {written} written "
                      f"[{time.time() - started:.0f}s]", flush=True)
        print(f"[vggish] {split}: wrote {written} ({skipped} cached, {missing} missing) "
              f"-> {out_dir} [{time.time() - started:.0f}s]", flush=True)
        total += written
    print(f"[vggish] done: {total} features in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
