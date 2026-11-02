#!/usr/bin/env python
"""Create binary CCTV lists for the SCFD (Surveillance Camera Fight Dataset).

SCFD contains surveillance videos of fights and non-fights.
videos.txt contains mappings of trimmed clips to source YouTube videos:
    https://www.youtube.com/watch?v=...
    fi001: ...
    nofi001: ...

To prevent data leakage, clips sharing the same source video must be placed in the
same split (source-disjoint partitioning).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safewatch.edge.extractor import SNIPPET_FRAMES  # noqa: E402
from safewatch.edge.video import probe  # noqa: E402


def parse_video_sources(videos_txt: Path) -> dict[str, str]:
    """Map clip stem (e.g. 'fi001', 'nofi001') to source URL/ID."""
    clip_to_source = {}
    current_source = "unknown"
    if not videos_txt.is_file():
        return clip_to_source

    with videos_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("http://") or line.startswith("https://"):
                current_source = line.split("v=")[-1] if "v=" in line else line
            elif ":" in line:
                clip_id = line.split(":")[0].strip()
                clip_to_source[clip_id] = current_source
    return clip_to_source


def build_scfd_rows(root: Path, val_ratio: float = 0.2, seed: int = 43) -> tuple[list[dict], list[dict], dict]:
    clip_to_source = parse_video_sources(root / "videos.txt")

    # Group video files by source
    fight_dir = root / "fight"
    nofight_dir = root / "noFight"

    all_clips = []
    # label 1: fight, label 0: nofight
    for video in sorted(fight_dir.glob("*.mp4")):
        stem = video.stem
        src = clip_to_source.get(stem, f"fight_{stem}")
        all_clips.append((video, 1, src, stem))

    for video in sorted(nofight_dir.glob("*.mp4")):
        stem = video.stem
        src = clip_to_source.get(stem, f"nofight_{stem}")
        all_clips.append((video, 0, src, stem))

    # Group by source to ensure source-disjoint split
    source_to_clips: dict[str, list] = {}
    for item in all_clips:
        src = item[2]
        source_to_clips.setdefault(src, []).append(item)

    rng = random.Random(seed)
    sources = sorted(source_to_clips.keys())
    rng.shuffle(sources)

    train_clips = []
    val_clips = []
    total_clips = len(all_clips)
    val_target = int(total_clips * val_ratio)

    val_count = 0
    for src in sources:
        items = source_to_clips[src]
        if val_count < val_target:
            val_clips.extend(items)
            val_count += len(items)
        else:
            train_clips.extend(items)

    def to_row(video: Path, label: int, src: str, stem: str, split: str) -> dict:
        clip_id = f"scfd_{split}_{'fight' if label else 'normal'}_{stem}"
        try:
            p = probe(video)
            frames = int(p.get("frames") or 0)
        except Exception:
            frames = 0
        if frames < 1:
            frames = SNIPPET_FRAMES  # fallback 1 snippet if probe fails

        return {
            "clip_id": clip_id,
            "split": split,
            "path": "",
            "video_path": str(video.relative_to(ROOT)),
            "T": str(max(1, frames // SNIPPET_FRAMES)),
            "binary": str(label),
            "labels": "fighting" if label else "",
            "dataset": "SCFD",
            "source_id": src,
        }

    train_rows = [to_row(v, lbl, s, stem, "train") for v, lbl, s, stem in train_clips]
    val_rows = [to_row(v, lbl, s, stem, "val") for v, lbl, s, stem in val_clips]

    stats = {
        "dataset": "SCFD",
        "total_clips": total_clips,
        "train": {
            "clips": len(train_rows),
            "violent": sum(r["binary"] == "1" for r in train_rows),
            "normal": sum(r["binary"] == "0" for r in train_rows),
        },
        "val": {
            "clips": len(val_rows),
            "violent": sum(r["binary"] == "1" for r in val_rows),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/cctv_datasets/scfd")
    parser.add_argument("--out", default="data/lists_scfd_binary")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    root = ROOT / args.root
    out = ROOT / args.out

    if not root.is_dir():
        raise SystemExit(f"[scfd] Root dir not found: {root}")

    train_rows, val_rows, stats = build_scfd_rows(root, val_ratio=args.val_ratio, seed=args.seed)
    write_csv(out / "train.csv", train_rows)
    write_csv(out / "val.csv", val_rows)
    (out / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

