#!/usr/bin/env python
"""Analyze false negatives from a frozen binary evaluation without changing the model.

This is a diagnostic, not another threshold search.  The supplied report must already contain the
threshold selected on validation; this script merely identifies the positive clips predicted as 0
and writes a compact aggregate report plus a CSV for manual video review.

Example:
    .venv/bin/python scripts/analyze_binary_false_negatives.py \
      --metrics runs/<run>/binary_source_disjoint_f1_validation_threshold_metrics.json \
      --predictions runs/<run>/binary_source_disjoint_f1_validation_threshold_predictions.npz \
      --test-csv data/lists_binary_minirgb_sourceholdout/test.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def source_of(clip_id: str) -> str:
    return clip_id.split("__#", 1)[0]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": round(float(values.min()), 4),
        "p25": round(float(np.percentile(values, 25)), 4),
        "median": round(float(np.median(values)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics["threshold"])
    stride_frames = float(metrics.get("stride_frames", 16.0))
    archive = np.load(args.predictions)
    ids = [str(value) for value in archive["clip_ids"]]
    labels = np.asarray(archive["labels"], dtype=np.int64)
    probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    if not (len(ids) == labels.size == probabilities.size):
        raise ValueError("predictions arrays have inconsistent lengths")
    rows = read_rows(args.test_csv)
    missing = sorted(set(ids) - set(rows))
    if missing:
        raise ValueError(f"{len(missing)} prediction IDs are absent from --test-csv")

    positive = labels == 1
    false_negative = positive & (probabilities < threshold)
    duration = np.asarray([float(rows[clip_id]["T"]) * stride_frames / 24.0 for clip_id in ids])
    margins = threshold - probabilities[false_negative]
    fn_indices = np.flatnonzero(false_negative)
    fn_indices = fn_indices[np.argsort(margins)]  # closest misses first: easiest review candidates

    review_rows = []
    for index in fn_indices:
        row = rows[ids[index]]
        review_rows.append({
            "clip_id": ids[index],
            "source": source_of(ids[index]),
            "clip_probability": round(float(probabilities[index]), 6),
            "threshold": round(threshold, 6),
            "margin_below_threshold": round(float(threshold - probabilities[index]), 6),
            "duration_seconds": round(float(duration[index]), 2),
            "feature_path": row["path"],
        })

    duration_bands = {}
    for upper in (30.0, 60.0, 120.0):
        all_count = int(np.sum(positive & (duration <= upper)))
        fn_count = int(np.sum(false_negative & (duration <= upper)))
        duration_bands[f"at_most_{upper:g}s"] = {
            "positive_clips": all_count,
            "false_negatives": fn_count,
            "false_negative_rate": round(fn_count / all_count, 4) if all_count else None,
        }
    source_counts = Counter(source_of(ids[index]) for index in fn_indices)
    report = {
        "purpose": "binary false-negative diagnostic; no threshold fitting or model update",
        "metrics": str(args.metrics),
        "predictions": str(args.predictions),
        "test_csv": str(args.test_csv),
        "threshold": threshold,
        "stride_frames": stride_frames,
        "n_scored": int(labels.size),
        "n_positive": int(positive.sum()),
        "n_false_negative": int(false_negative.sum()),
        "false_negative_rate": round(float(false_negative.sum() / positive.sum()), 4),
        "positive_probability": describe(probabilities[positive]),
        "false_negative_probability": describe(probabilities[false_negative]),
        "positive_duration_seconds": describe(duration[positive]),
        "false_negative_duration_seconds": describe(duration[false_negative]),
        "duration_bands": duration_bands,
        "sources_with_multiple_false_negatives": [
            {"source": source, "false_negatives": count}
            for source, count in source_counts.most_common() if count > 1
        ],
        "review_order": "ascending margin below threshold (closest misses first)",
        "review_csv": "false_negatives_review.csv",
        "interpretation": [
            "A duration/source association is diagnostic only; review videos before assigning "
            "a cause.",
            "Closest misses are candidates for a validation-only threshold sensitivity check.",
            "Large-margin misses are candidates for representation, data, or temporal-model "
            "improvement.",
        ],
    }
    out_dir = args.out_dir or args.metrics.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "binary_false_negative_analysis.json"
    review_path = out_dir / "false_negatives_review.csv"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(review_rows[0]) if review_rows else ["clip_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    print(json.dumps(report, indent=2))
    print(f"[ok] false-negative report -> {report_path}")
    print(f"[ok] manual review list -> {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
