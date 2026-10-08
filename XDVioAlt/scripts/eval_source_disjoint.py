#!/usr/bin/env python
"""Re-score a run's saved predictions on the **source-disjoint** subset of the test split.

The official XD-Violence split is random at clip level, so a large share of test clips come from a
source (film / YouTube video) that also appears in training (measured: 59 % of the 800 test clips).
AP on those clips partly measures "does the model recognise this film", not "does it detect the
event". This script recomputes the metrics on the subset whose source is absent from training, using
the predictions already saved by ``safewatch.eval.runner`` (no model re-run needed).

Usage:
    .venv/bin/python scripts/eval_source_disjoint.py \
        --predictions runs/2026-09-23_p1_visual_swin/predictions.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, ".")

from build_lists import source_of  # noqa: E402

from safewatch.eval import metrics as M  # noqa: E402


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def metrics_for(ids: list[str], scores: dict, gt: dict) -> dict:
    pred = {k: scores[k] for k in ids}
    truth = {k: gt[k] for k in ids}
    row: dict = {}
    for protocol in ("global", "per_video"):
        for metric in ("pr_auc", "average_precision"):
            row[f"{protocol}__{metric}"] = M.evaluate(pred, truth, protocol=protocol,
                                                      metric=metric)["ap"]
    row["prevalence"] = M.prevalence(np.concatenate([truth[k] for k in sorted(truth)]))
    row["n_clips"] = len(ids)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="predictions.npz from the runner")
    parser.add_argument("--train-csv", default="data/lists/train.csv")
    parser.add_argument("--test-csv", default="data/lists/test.csv")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    train_sources = {source_of(r["clip_id"]) for r in read_csv(args.train_csv)}
    test_rows = read_csv(args.test_csv)

    archive = np.load(args.predictions)
    scores = {k[len("scores::"):]: archive[k] for k in archive.files if k.startswith("scores::")}
    gt = {k[len("gt::"):]: archive[k] for k in archive.files if k.startswith("gt::")}

    seen = [r["clip_id"] for r in test_rows
            if r["clip_id"] in scores and source_of(r["clip_id"]) in train_sources]
    unseen = [r["clip_id"] for r in test_rows
              if r["clip_id"] in scores and source_of(r["clip_id"]) not in train_sources]

    report = {
        "predictions": str(args.predictions),
        "n_test_scored": len(scores),
        "n_source_seen": len(seen),
        "n_source_unseen": len(unseen),
        "share_source_seen_pct": round(100 * len(seen) / max(1, len(scores)), 1),
        "all": metrics_for(sorted(scores), scores, gt),
        "source_seen": metrics_for(seen, scores, gt),
        "source_unseen": metrics_for(unseen, scores, gt),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
