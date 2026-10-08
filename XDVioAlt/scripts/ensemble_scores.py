#!/usr/bin/env python
"""Average the snippet-score curves of several runs into one ensemble prediction set.

The fusion variants (``late`` / ``cross`` / ``early``) are trained on identical data
with identical hyper-parameters and no gating, but they make *different* errors, so averaging their
score curves buys accuracy for free - no new backbone, no GPU, no extra features. Measured on the
mirror grid (800 test clips):

    late                        0.8125 global AP / 0.7621 per-clip
    late+cross+early ensemble   0.8216 global AP / 0.7652 per-clip
    delta                       +0.0091  bootstrap 95% CI [+0.0053, +0.0138], P(delta>0)=1.000

``adaptive`` is deliberately *not* a member: it is the weakest and least seed-stable variant
(``docs/journal.md``), and including it drags the average down (0.8123).

Edge cost is small: the members share the same input features, so deployment extracts features once
and runs N tiny heads (6.5 ms each - ``docs/edge_ai.md``).

Usage:
    .venv/bin/python scripts/ensemble_scores.py \
        --runs runs/*_p3full_late runs/*_p3full_cross runs/*_p3full_early \
        --out runs/2026-10-04_p3full_ens_lce
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.eval import metrics as M  # noqa: E402
from safewatch.eval.runner import (  # noqa: E402
    STRIDE_FRAMES,
    build_gt,
    compute_metrics,
    load_intervals,
    load_rows,
)


def load_scores(run: Path) -> dict[str, np.ndarray]:
    """Snippet score curves from a run's ``predictions.npz`` -> {clip_id: (T,)}."""
    path = run / "predictions.npz"
    if not path.exists():
        raise SystemExit(f"[ensemble] no predictions at {path}")
    data = np.load(path, allow_pickle=True)
    return {key.split("::", 1)[1]: data[key] for key in data.files if key.startswith("scores::")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run dirs, each containing predictions.npz")
    parser.add_argument("--list", default=None, help="test csv (default: <lists_dir>/test.csv)")
    parser.add_argument("--intervals", default="data/annotations/test_intervals.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-seconds", type=float, default=None)
    parser.add_argument("--max-gap-seconds", type=float, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = [Path(r) for r in args.runs]
    cfg = json.loads((runs[0] / "config.json").read_text(encoding="utf-8"))
    stride = STRIDE_FRAMES[str(cfg.get("feature_set") or "swin_rgb")]
    list_path = Path(args.list or Path(cfg.get("lists_dir", "data/lists")) / "test.csv")

    members = [load_scores(run) for run in runs]
    common = sorted(set.intersection(*[set(m) for m in members]))
    if not common:
        raise SystemExit("[ensemble] the runs share no clip ids")
    print(f"[ensemble] {len(runs)} members, {len(common)} common clips, stride {stride:g}")

    scores = {cid: np.mean([m[cid] for m in members], axis=0).astype(np.float32)
              for cid in common}

    keep = set(common)
    rows = [r for r in load_rows(list_path) if r["clip_id"] in keep]
    intervals = load_intervals(args.intervals)
    gt = build_gt(rows, intervals, stride)
    gt_segments = {cid: M.gt_segments_from_intervals(intervals.get(cid, []), stride)
                   for cid in scores}

    min_length = (M.snippets_for_seconds(args.min_seconds, stride, args.fps)
                  if args.min_seconds is not None else 2)
    max_gap = (M.snippets_for_seconds(args.max_gap_seconds, stride, args.fps, minimum=0)
               if args.max_gap_seconds is not None else 1)
    multilogits = {cid: np.zeros(6, dtype=np.float32) for cid in common}
    report = compute_metrics(scores, gt, gt_segments, multilogits, rows, stride,
                             args.threshold, min_length, max_gap, fps=args.fps)
    # the scorer does not persist per-class logits, so categories cannot be ensembled: drop the fake
    # per-class block rather than publish an all-zeros category result
    report.pop("per_class_clip_level_ap", None)
    report.pop("macro_per_class_ap", None)
    report["category_note"] = "not ensembled: per-class logits are not persisted by the scorer"
    report["ensemble_members"] = [run.name for run in runs]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz",
                        **{f"scores::{k}": v for k, v in scores.items()},
                        **{f"gt::{k}": v for k, v in gt.items()})
    (out / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ensemble] global AP {report['ap']['global__pr_auc']:.4f}  "
          f"per-clip {report['ap']['per_video__pr_auc']:.4f}  "
          f"F1@IoU.5 {report['localization']['f1@iou0.5']:.4f}")
    print(f"[ensemble] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
