#!/usr/bin/env python3
"""Prepare binary dataset splits for the smart-dashcam accident/driving dataset.

Contains:
- 68 accident videos (binary=1, labels='car_accident')
- 68 normal driving videos (binary=0, labels='')
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safewatch.edge.extractor import SNIPPET_FRAMES  # noqa: E402
from safewatch.edge.video import probe  # noqa: E402


def build_dashcam_rows(data_dir: Path, meta_csv: Path, val_ratio: float = 0.2, seed: int = 43) -> tuple[list[dict], list[dict], dict]:
    with meta_csv.open("r", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

    accident_clips = []
    normal_clips = []

    for r in meta_rows:
        fn = r["filename"]
        case = r["case"].strip().lower()
        v_path = data_dir / fn
        if not v_path.is_file():
            # sometimes files are unpacked under a subfolder
            candidates = list(data_dir.glob(f"**/{fn}"))
            if candidates:
                v_path = candidates[0]
            else:
                continue

        is_accident = (case == "accident")
        item = (v_path, 1 if is_accident else 0, fn)
        if is_accident:
            accident_clips.append(item)
        else:
            normal_clips.append(item)

    rng = random.Random(seed)
    rng.shuffle(accident_clips)
    rng.shuffle(normal_clips)

    val_acc_n = max(1, int(len(accident_clips) * val_ratio))
    val_norm_n = max(1, int(len(normal_clips) * val_ratio))

    val_clips = accident_clips[:val_acc_n] + normal_clips[:val_norm_n]
    train_clips = accident_clips[val_acc_n:] + normal_clips[val_norm_n:]

    def to_row(v_path: Path, label: int, fn: str, split: str) -> dict:
        stem = Path(fn).stem
        clip_id = f"dashcam_{split}_{'accident' if label else 'normal'}_{stem}"
        try:
            info = probe(v_path)
            frames = int(info.get("frames") or 0)
        except Exception:
            frames = 0
        if frames <= 0:
            frames = SNIPPET_FRAMES * 5

        return {
            "clip_id": clip_id,
            "split": split,
            "path": "",
            "video_path": str(v_path.relative_to(ROOT)),
            "T": str(max(1, frames // SNIPPET_FRAMES)),
            "binary": str(label),
            "labels": "car_accident" if label else "",
            "dataset": "SmartDashcam",
        }

    train_rows = [to_row(v, lbl, fn, "train") for v, lbl, fn in train_clips]
    val_rows = [to_row(v, lbl, fn, "val") for v, lbl, fn in val_clips]

    stats = {
        "dataset": "SmartDashcam",
        "total": len(train_rows) + len(val_rows),
        "train": {
            "clips": len(train_rows),
            "accidents": sum(r["binary"] == "1" for r in train_rows),
            "normal": sum(r["binary"] == "0" for r in train_rows),
        },
        "val": {
            "clips": len(val_rows),
            "accidents": sum(r["binary"] == "1" for r in val_rows),
            "normal": sum(r["binary"] == "0" for r in val_rows),
        },
    }
    return train_rows, val_rows, stats


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    data_dir = ROOT / "data/raw/dashcam_accidents"
    meta_csv = data_dir / "train.csv"
    out_dir = ROOT / "data/lists_dashcam_binary"

    if not meta_csv.is_file():
        raise SystemExit(f"Metadata CSV not found: {meta_csv}")

    train_rows, val_rows, stats = build_dashcam_rows(data_dir, meta_csv)
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)
    (out_dir / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

