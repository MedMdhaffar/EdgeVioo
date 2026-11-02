#!/usr/bin/env python3
"""Export only human-reviewed UCA intervals from the UCA train split.

Blank labels, ``ignore``, and all UCA val/test rows are excluded. This ensures that caption
keyword matches and unannotated time cannot silently become training targets or negatives.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"fighting", "shooting", "riot", "abuse", "car_accident", "explosion"}
VALID_LABELS = CLASSES | {"normal", "ignore"}
OUTPUT_FIELDS = ["video_name", "video_path", "start_sec", "end_sec", "duration_sec",
                 "review_label", "target_binary", "description", "source_file"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/annotations/uca_intervals_review.csv")
    parser.add_argument("--output", default="data/annotations/uca_intervals_supervised_train.csv")
    args = parser.parse_args()
    source, output = ROOT / args.input, ROOT / args.output
    if not source.is_file():
        parser.error(f"review table not found: {source}")

    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    video_splits: dict[str, str] = {}
    for row in csv.DictReader(source.open(newline="", encoding="utf-8")):
        video = row["video_name"]
        split = row["split"].lower()
        if video in video_splits and video_splits[video] != split:
            raise ValueError(f"video appears in more than one UCA split: {video}")
        video_splits[video] = split
        counts[f"source_{split}_rows"] += 1
        if split != "train":
            continue
        label = (row.get("review_label") or "").strip().lower()
        if not label:
            counts["train_unreviewed"] += 1
            continue
        if label not in VALID_LABELS:
            raise ValueError(f"invalid review_label {label!r} for {video}; expected one of {sorted(VALID_LABELS)}")
        if label == "ignore":
            counts["train_ignored"] += 1
            continue
        start, end, duration = float(row["start_sec"]), float(row["end_sec"]), float(row["duration_sec"])
        if start < 0 or end <= start or (duration and end > duration + 0.25):
            raise ValueError(f"invalid interval [{start}, {end}] (duration {duration}) in {video}")
        rows.append({
            "video_name": video,
            "video_path": row["video_path"],
            "start_sec": f"{start:.3f}",
            "end_sec": f"{end:.3f}",
            "duration_sec": f"{duration:.3f}",
            "review_label": label,
            "target_binary": "0" if label == "normal" else "1",
            "description": row["description"],
            "source_file": row["source_file"],
        })
        counts[f"train_label_{label}"] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input": str(source.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "exported_reviewed_train_intervals": len(rows),
        "counts": dict(counts),
        "binary_target": "1 for any of the six reviewed violence classes; 0 only for explicitly reviewed normal intervals",
        "unknown_policy": "Blank and ignored labels are excluded. Unannotated time is not exported. UCA val/test rows are never exported for training.",
        "video_split_validation": "Each video name was checked to occur in only one UCA split.",
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
