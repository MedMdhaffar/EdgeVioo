#!/usr/bin/env python
"""Per-category localisation report from a run's saved predictions (no model re-run).

Why this exists: ``test_metrics.json`` reports localisation **binary** (one incident per clip), but
a supervision tool has to show *which* category is where. The XD-Violence interval annotations carry
no category column (``clip_id,start,end``), so per-category localisation is defined here as
*localisation restricted to the clips whose weak label set contains that category*. That is what a
per-category alert would be scored on, and it exposes categories whose incidents are found in the
wrong place even when their clip-level AP looks healthy.

The segment rule is expressed in **seconds** and converted with the measured grid
(:func:`safewatch.eval.metrics.snippets_for_seconds`), because the same snippet count means
different durations on the mirror (64 frames/snippet) and official (16) grids - the trap documented
in ``docs/journal.md`` Session 10.

Usage:
    .venv/bin/python scripts/per_class_localization.py \\
        --predictions runs/2026-09-29_p3vggish30ep_cross/predictions.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.data.dataset import CATEGORIES  # noqa: E402
from safewatch.eval import metrics as M  # noqa: E402

DEFAULT_MIN_SECONDS = 5.333   # the mirror-grid reference rule (2 snippets x 64 frames / 24 fps)
DEFAULT_MAX_GAP_SECONDS = 2.667


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_intervals(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {}
    for row in read_rows(path):
        intervals.setdefault(row["clip_id"], []).append((int(row["start"]), int(row["end"])))
    return intervals


def labels_of(row: dict) -> set[str]:
    return {c for c in (row.get("labels") or "").split("|") if c}


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {k[len("scores::"):]: archive[k] for k in archive.files if k.startswith("scores::")}


def resolve_stride(predictions: str | Path, explicit: float | None, metrics: str | None) -> float:
    """Grid used for frame-interval conversion: explicit flag > sibling metrics file."""
    if explicit:
        return float(explicit)
    candidate = Path(metrics) if metrics else Path(predictions).parent / "test_metrics.json"
    if candidate.exists():
        stored = json.loads(candidate.read_text(encoding="utf-8")).get("stride_frames")
        if stored:
            return float(stored)
    raise SystemExit("[per-class] cannot infer the grid: pass --stride-frames "
                     f"(looked at {candidate})")


def report_for(ids: list[str], scores: dict, intervals: dict, stride: float, threshold: float,
               min_length: int, max_gap: int, fps: float) -> dict:
    """Localisation report restricted to ``ids`` (segments in snippets, durations in seconds)."""
    pred = {cid: M.segments_from_scores(scores[cid], threshold, min_length, max_gap) for cid in ids}
    gt = {cid: M.gt_segments_from_intervals(intervals.get(cid, []), stride) for cid in ids}
    return M.localization_report(pred, gt, stride=stride, fps=fps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="predictions.npz from the runner")
    parser.add_argument("--test-csv", default=None,
                        help="default: the lists dir implied by the checkpoint's config.json")
    parser.add_argument("--intervals", default="data/annotations/test_intervals.csv")
    parser.add_argument("--metrics", default=None,
                        help="test_metrics.json (default: next to --predictions)")
    parser.add_argument("--stride-frames", type=float, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    stride = resolve_stride(args.predictions, args.stride_frames, args.metrics)
    test_csv = args.test_csv
    if test_csv is None:
        config = Path(args.predictions).parent / "config.json"
        lists_dir = "data/lists"
        if config.exists():
            lists_dir = json.loads(config.read_text(encoding="utf-8")).get("lists_dir") or lists_dir
        test_csv = str(Path(lists_dir) / "test.csv")

    min_length = M.snippets_for_seconds(args.min_seconds, stride, args.fps)
    max_gap = M.snippets_for_seconds(args.max_gap_seconds, stride, args.fps, minimum=0)
    scores = load_predictions(args.predictions)
    rows = [r for r in read_rows(test_csv) if r["clip_id"] in scores]
    intervals = read_intervals(args.intervals)

    print(f"[per-class] {args.predictions} clips={len(rows)} stride={stride:g} "
          f"rule=({min_length},{max_gap}) snippets = "
          f"({args.min_seconds:g},{args.max_gap_seconds:g}) s")

    out: dict = {
        "predictions": str(args.predictions),
        "test_csv": str(test_csv),
        "stride_frames": stride,
        "segment_rule": {"threshold": args.threshold, "min_length": min_length, "max_gap": max_gap,
                         "min_seconds": args.min_seconds, "max_gap_seconds": args.max_gap_seconds},
        "overall": report_for([r["clip_id"] for r in rows], scores, intervals, stride,
                              args.threshold, min_length, max_gap, args.fps),
        "per_category": {},
    }

    header = (f"{'category':14s} {'clips':>5s} {'gtseg':>5s} {'predseg':>7s} {'IoU':>5s} "
              f"{'F1@.5':>6s} {'delay':>7s}")
    print(header)
    print("-" * len(header))
    for category in CATEGORIES:
        ids = [r["clip_id"] for r in rows if category in labels_of(r)]
        if not ids:
            continue
        rep = report_for(ids, scores, intervals, stride, args.threshold,
                         min_length, max_gap, args.fps)
        out["per_category"][category] = {"n_clips": len(ids), **rep}
        print(f"{category:14s} {len(ids):5d} {rep['n_gt_segments']:5d} {rep['n_pred_segments']:7d} "
              f"{rep['mean_best_iou']:5.3f} {rep['f1@iou0.5']:6.3f} "
              f"{rep['mean_detection_delay_seconds']:6.1f}s")
    overall = out["overall"]
    print(f"{'ALL':14s} {len(rows):5d} {overall['n_gt_segments']:5d} "
          f"{overall['n_pred_segments']:7d} {overall['mean_best_iou']:5.3f} "
          f"{overall['f1@iou0.5']:6.3f} {overall['mean_detection_delay_seconds']:6.1f}s")
    print("legend: per-category = localisation restricted to clips whose weak labels contain it; "
          "delay = predicted start minus matched-GT start (negative = early)")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[per-class] written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
