#!/usr/bin/env python
"""Create binary CCTV lists for the extracted RWF-2000 dataset.

RWF-2000 has a vendor-defined train/validation split and direct fight/non-fight labels.  This
script preserves it, prefixes clip IDs so its audio/features cannot collide with XD-Violence, and
keeps the original ``.avi`` location in ``video_path`` for the extractor scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safewatch.edge.extractor import SNIPPET_FRAMES  # noqa: E402
from safewatch.edge.video import probe  # noqa: E402

SPLITS = {
    "train": (("train/Train_Fight", 1), ("train/Train_NonFight", 0)),
    "val": (("val/Val_Fight", 1), ("val/Val_NonFight", 0)),
}


import hashlib

def sanitize_clip_id(split: str, label: int, stem: str) -> str:
    lbl_str = 'fight' if label else 'normal'
    raw_id = f"rwf_{split}_{lbl_str}_{stem}"
    if len(raw_id.encode("utf-8")) > 120 or not raw_id.isascii():
        h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:12]
        return f"rwf_{split}_{lbl_str}_safe_{h}"
    return raw_id


def rows_for(root: Path, split: str) -> list[dict[str, str]]:
    rows = []
    for relative, label in SPLITS[split]:
        for video in sorted((root / relative).glob("*.avi")):
            clip_id = sanitize_clip_id(split, label, video.stem)
            frame_count = int(probe(video).get("frames") or 0)
            if frame_count < 1:
                raise RuntimeError(f"[rwf] cannot determine frame count: {video}")
            rows.append({
                "clip_id": clip_id,
                "split": split,
                "path": "",
                "video_path": str(video.relative_to(ROOT)),
                "T": str(max(1, frame_count // SNIPPET_FRAMES)),
                "binary": str(label),
                "labels": "fighting" if label else "",
                "dataset": "RWF-2000",
            })
    return rows


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/cctv_datasets/rwf2000/RWF-2000")
    parser.add_argument("--out", default="data/lists_rwf2000_binary")
    args = parser.parse_args()
    root, out = ROOT / args.root, ROOT / args.out
    if not root.is_dir():
        raise SystemExit(f"[rwf] dataset root not found: {root}")
    summary = {"dataset": "RWF-2000", "task": "binary violence (fight=1, non-fight=0)",
               "source_root": str(root.relative_to(ROOT)), "splits": {}}
    for split in SPLITS:
        rows = rows_for(root, split)
        if not rows:
            raise SystemExit(f"[rwf] no AVI files found for {split} under {root}")
        write(out / f"{split}.csv", rows)
        summary["splits"][split] = {"clips": len(rows),
                                    "violent": sum(r["binary"] == "1" for r in rows),
                                    "normal": sum(r["binary"] == "0" for r in rows)}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
