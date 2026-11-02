#!/usr/bin/env python3
"""Synthesize augmented normal driving training samples by:
1. Temporal sliding-window slicing on existing normal driving recordings (188 samples)
2. Mining authentic pre-impact driving windows from accident recordings (316 samples)

Total new diverse normal driving samples: ~500, balancing dashcam vs CCTV.
"""
from __future__ import annotations

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


def main() -> int:
    train_csv = ROOT / "data/lists_joint_all_k4_mstd/train.csv"
    rows = load_rows(train_csv)
    
    out_dir = ROOT / "data/features/minirgb_k4_mstd/train_augmented_normal"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mel_out_dir = ROOT / "data/features/mel/train_augmented_normal"
    mel_out_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    
    # 1. Slide windows across existing normal driving clips
    normal_dash = [r for r in rows if r.get("dataset") == "SmartDashcam" and str(r.get("binary")) == "0"]
    for r in normal_dash:
        cid = r["clip_id"]
        feat_path = ROOT / r["path"]
        if not feat_path.exists():
            continue
        feat = np.load(feat_path)
        t = feat.shape[0]
        window_size = 2
        for start_idx in range(0, max(1, t - window_size + 1)):
            sub_feat = feat[start_idx:start_idx + window_size]
            sub_cid = f"{cid}_win{start_idx}"
            sub_path = out_dir / f"{sub_cid}.npy"
            save_array(sub_path, sub_feat)
            
            # create matching zero audio
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

    # 2. Mine pre-impact segments from accident clips
    acc_dash = [r for r in rows if r.get("dataset") == "SmartDashcam" and str(r.get("binary")) == "1"]
    for r in acc_dash:
        cid = r["clip_id"]
        feat_path = ROOT / r["path"]
        if not feat_path.exists():
            continue
        feat = np.load(feat_path)
        t = feat.shape[0]
        if t >= 6:
            # First t - 2 snippets are before collision
            pre_t = t - 2
            window_size = 2
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

    print(f"[augmented] Created {len(new_rows)} new authentic normal driving training samples!")
    
    # Write augmented joint list
    augmented_train_csv = ROOT / "data/lists_joint_all_k4_mstd/train_augmented.csv"
    all_train_rows = rows + new_rows
    with augmented_train_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["clip_id", "split", "path", "T", "binary", "labels", "dataset", "video_path", "source_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_train_rows)

    print(f"[augmented] Saved {len(all_train_rows)} total training clips to {augmented_train_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

