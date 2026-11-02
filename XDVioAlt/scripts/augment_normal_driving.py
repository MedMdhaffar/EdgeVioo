#!/usr/bin/env python3
"""Synthesize augmented normal driving training samples by:
1. Temporal sliding-window slicing on existing normal driving recordings (188 samples)
2. Mining authentic pre-impact driving windows from accident recordings (316 samples)

Total new diverse normal driving samples: ~500, balancing dashcam vs CCTV.

Pre-crash guard (--precrash-guard N, default 3):
    At k4_mstd each snippet covers ~4 frames × 0.167 s ≈ 0.67 s. The guard discards the
    last N snippets of every accident clip before mining pre-impact normals, ensuring that
    the mined windows are safely before the impact region. The default (3 snippets ≈ 2 s)
    is conservative relative to the original hard-coded value of 2 snippets (≈ 1.3 s).
    Increase for faster cameras or longer impact build-ups; the clip is skipped entirely
    if t < guard + window_size to prevent mining from within the collision itself.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safewatch.data.dataset import load_rows


def save_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", default="data/lists_joint_all_k4_mstd/train.csv",
                        help="Source training CSV (default: %(default)s)")
    parser.add_argument("--out-csv", default="data/lists_joint_all_k4_mstd/train_augmented.csv",
                        help="Output augmented CSV (default: %(default)s)")
    parser.add_argument("--window-size", type=int, default=2,
                        help="Snippet window size for each synthetic clip (default: %(default)s)")
    parser.add_argument("--precrash-guard", type=int, default=3,
                        help="Snippets to exclude from the end of accident clips when mining "
                             "pre-impact normals. At k4_mstd, 1 snippet ≈ 0.67 s. "
                             "Default %(default)s (≈ 2 s). Original hard-coded value was 2.")
    args = parser.parse_args(argv)

    train_csv = ROOT / args.train_csv
    window_size = args.window_size
    guard = args.precrash_guard

    rows = load_rows(train_csv)

    out_dir = ROOT / "data/features/minirgb_k4_mstd/train_augmented_normal"
    out_dir.mkdir(parents=True, exist_ok=True)

    mel_out_dir = ROOT / "data/features/mel/train_augmented_normal"
    mel_out_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []

    # 1. Slide windows across existing normal driving clips
    normal_dash = [r for r in rows
                   if r.get("dataset") == "SmartDashcam" and str(r.get("binary")) == "0"]
    for r in normal_dash:
        cid = r["clip_id"]
        feat_path = ROOT / r["path"]
        if not feat_path.exists():
            continue
        feat = np.load(feat_path)
        t = feat.shape[0]
        for start_idx in range(0, max(1, t - window_size + 1)):
            sub_feat = feat[start_idx:start_idx + window_size]
            sub_cid = f"{cid}_win{start_idx}"
            sub_path = out_dir / f"{sub_cid}.npy"
            save_array(sub_path, sub_feat)

            # create matching zero audio (normal dashcam has no event audio)
            sub_mel = np.zeros((window_size, 132), dtype=np.float32)
            save_array(mel_out_dir / f"{sub_cid}.npy", sub_mel)

            new_rows.append({
                "clip_id": sub_cid,
                "split": "train",
                "path": str(sub_path.relative_to(ROOT)),
                "T": str(window_size),
                "binary": "0",
                "labels": "",
                "dataset": "SmartDashcam",
                "video_path": r.get("video_path", ""),
                "source_id": "dashcam_normal_slice",
            })

    # 2. Mine pre-impact segments from accident clips.
    # The guard excludes the last `guard` snippets (≈ guard × 0.67 s at k4_mstd) to keep
    # all mined windows safely before the collision region. Clips shorter than
    # guard + window_size are skipped entirely to avoid any overlap with the impact.
    acc_dash = [r for r in rows
                if r.get("dataset") == "SmartDashcam" and str(r.get("binary")) == "1"]
    skipped_short = 0
    for r in acc_dash:
        cid = r["clip_id"]
        feat_path = ROOT / r["path"]
        if not feat_path.exists():
            continue
        feat = np.load(feat_path)
        t = feat.shape[0]
        pre_t = t - guard  # safe pre-impact region length
        if pre_t < window_size:
            # Clip too short to mine any window outside the guarded region
            skipped_short += 1
            continue
        for start_idx in range(0, max(1, pre_t - window_size + 1)):
            sub_feat = feat[start_idx:start_idx + window_size]
            sub_cid = f"{cid}_precrash_win{start_idx}"
            sub_path = out_dir / f"{sub_cid}.npy"
            save_array(sub_path, sub_feat)

            sub_mel = np.zeros((window_size, 132), dtype=np.float32)
            save_array(mel_out_dir / f"{sub_cid}.npy", sub_mel)

            new_rows.append({
                "clip_id": sub_cid,
                "split": "train",
                "path": str(sub_path.relative_to(ROOT)),
                "T": str(window_size),
                "binary": "0",
                "labels": "",
                "dataset": "SmartDashcam",
                "video_path": r.get("video_path", ""),
                "source_id": "dashcam_precrash_normal",
            })

    print(f"[augmented] guard={guard} snippets (~{guard * 0.67:.1f} s at k4_mstd), "
          f"window={window_size} snippets, skipped {skipped_short} short accident clips")
    print(f"[augmented] Created {len(new_rows)} new authentic normal driving training samples!")

    # Write augmented joint list
    augmented_train_csv = ROOT / args.out_csv
    all_train_rows = rows + new_rows
    with augmented_train_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["clip_id", "split", "path", "T", "binary", "labels",
                      "dataset", "video_path", "source_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_train_rows)

    print(f"[augmented] Saved {len(all_train_rows)} total training clips to {augmented_train_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

