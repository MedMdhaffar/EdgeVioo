#!/usr/bin/env python
"""Bridge the official XD-Violence VGGish features into the layout our dataset expects.

Why this exists
---------------
The journal (Session 9) records the audio bottleneck as a *representation* problem: our audio stream
is 132-d log-mel patch statistics hand-extracted with ffmpeg, while the XD-Violence paper
128-d - and our own extraction covered 10 clips for lack of CPU budget. The authors' own VGGish
release is however already on disk at
``data/features/vggish_official_raw/vggish-features/{train,test}/<clip>__vggish.npy`` (4 754 clips,
581 MB, no download, no extraction). It only uses a different file name and a split subdirectory,
which is what this script adapts.

It also AUDITS the time alignment before linking anything: the official VGGish rows are meant to be
one-per-snippet on the official 16-frame I3D grid, so the row count must equal the ``T`` column of
the clip list. If it did not, feeding these features to the model would shift audio against video by
seconds and every fusion number would be meaningless - so a mismatch above a small tolerance aborts.

The link farm is a real directory of symlinks (np.load and Path.exists behave normally on it), and
nothing outside it is touched.

Usage:
    .venv/bin/python scripts/bridge_vggish.py                # audit + link train/val/test
    .venv/bin/python scripts/bridge_vggish.py --check-only   # audit, write nothing
    .venv/bin/python scripts/audio_stats.py --audio-dir data/features/vggish_snippet \\
        --train-csv data/lists_official/train.csv --audio-grid snippet \\
        --out data/features/vggish_stats.json                # then compute train stats
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUFFIX = "__vggish.npy"


def official_files(src: Path) -> dict:
    """clip_id -> npy path, across the split subdirectories of the official release."""
    index: dict[str, Path] = {}
    for path in sorted(src.rglob("*.npy")):
        name = path.name
        clip_id = name[:-len(SUFFIX)] if name.endswith(SUFFIX) else path.stem
        if clip_id in index:
            raise ValueError(f"duplicate clip_id in the official release: {clip_id}")
        index[clip_id] = path
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/features/vggish_official_raw/vggish-features")
    parser.add_argument("--out", default="data/features/vggish_snippet")
    parser.add_argument("--lists-dir", default="data/lists_official")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="max share of clips whose row count differs from T before aborting")
    parser.add_argument("--check-only", action="store_true", help="audit without writing links")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        print(f"[vggish] missing {src} - the official release is not on disk")
        return 1
    index = official_files(src)
    print(f"[vggish] {len(index)} official feature files under {src}")

    rows: list = []
    for split in ("train", "val", "test"):
        csv_path = Path(args.lists_dir) / f"{split}.csv"
        if csv_path.exists():
            rows.extend(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        print(f"[vggish] no clips found under {args.lists_dir}")
        return 1

    missing, mismatched, linked, shapes = [], [], 0, Counter()
    worst = 0.0
    for row in rows:
        clip_id, target_t = row["clip_id"], int(row["T"])
        source = index.get(clip_id)
        if source is None:
            missing.append(clip_id)
            continue
        rows_found = int(np.load(source, mmap_mode="r").shape[0])
        shapes[rows_found - target_t] += 1
        if rows_found != target_t:
            mismatched.append((clip_id, target_t, rows_found))
            worst = max(worst, abs(rows_found - target_t) / max(1, target_t))
        if not args.check_only:
            out.mkdir(parents=True, exist_ok=True)
            link = out / f"{clip_id}.npy"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(source, out))  # target is relative to the link dir
            linked += 1

    audited = sum(shapes.values())
    drift = sum(count for delta, count in shapes.items() if delta != 0)
    share = drift / max(1, audited)
    print(f"[vggish] audited {audited}/{len(rows)} clips: {drift} off-grid "
          f"({share:.2%}), worst relative drift {worst:.3f}")
    print(f"[vggish] (rows - T) histogram: {dict(sorted(shapes.items())[:6])}"
          + (" ..." if len(shapes) > 6 else ""))
    print("[vggish] missing features: "
          f"{len(missing)} clips" + (f" e.g. {missing[:2]}" if missing else ""))
    if not args.check_only:
        print(f"[vggish] {linked} symlinks -> {out}")

    if missing:
        print("[vggish] ABORT: clip list and feature release do not cover the same clips")
        return 1
    if share > args.tolerance:
        print(f"[vggish] ABORT: {share:.2%} of clips are off the snippet grid "
              f"(tolerance {args.tolerance:.2%}); results would be misaligned noise")
        return 1
    if mismatched:
        print(f"[vggish] note: {len(mismatched)} clips differ by a row or two; "
              "align_audio_to_snippets resamples those (nearest neighbour, +-1 snippet)")
    print("[vggish] OK - next: scripts/audio_stats.py --audio-grid snippet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
