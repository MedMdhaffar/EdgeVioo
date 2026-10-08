#!/usr/bin/env python
"""Per-category score diagnostic: is a weak class a *ranking* problem or a *label* problem?

``test_metrics.json`` gives one AP per category, which is not enough to act on. This script scores
the test split once and, for each category, reports:

* the score distribution of the positive clips (how many are above 0.5),
* the top-scoring **negative** clips, with their own label sets - if the false positives carry a
  *related* category, the weak AP is at least partly label ambiguity, not a modelling failure,
* the sub-category breakdown of each positive (mixed clips dilute the signal).

It also cross-references ``per_class_localization.json`` when present: a category whose incidents
are found in the right place (good tIoU) but whose ranking is poor is a **corpus-level** problem
(its clips look like other clips), not a temporal one. Used in ``docs/journal.md`` Session 10 to
explain ``abuse`` (AP 0.14 but tIoU 0.46).

Usage:
    .venv/bin/python scripts/class_diagnostic.py --ckpt runs/<run>/ckpt_best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.data.dataset import CATEGORIES  # noqa: E402
from safewatch.eval.runner import default_test_csv, score_clips  # noqa: E402


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def diagnostic(rows: list[dict], multilogits: dict[str, np.ndarray], category: str,
               top: int = 8) -> dict:
    index = CATEGORIES.index(category)
    prob = {r["clip_id"]: float(multilogits[r["clip_id"]][index]) for r in rows}
    positives = [r for r in rows if category in (r["labels"] or "").split("|")]
    negatives = [r for r in rows if category not in (r["labels"] or "").split("|")]

    pos_scores = np.array([prob[r["clip_id"]] for r in positives])
    neg_scores = np.array([prob[r["clip_id"]] for r in negatives])
    worst = sorted(positives, key=lambda r: prob[r["clip_id"]])[:top]
    best_neg = sorted(negatives, key=lambda r: -prob[r["clip_id"]])[:top]

    # how often does each other category co-occur with this one in the positives?
    co: dict[str, int] = {c: 0 for c in CATEGORIES}
    for r in positives:
        for c in (r["labels"] or "").split("|"):
            if c in co:
                co[c] += 1
    return {
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "mean_prob_positive": float(pos_scores.mean()) if pos_scores.size else float("nan"),
        "mean_prob_negative": float(neg_scores.mean()) if neg_scores.size else float("nan"),
        "n_positive_above_0.5": int((pos_scores >= 0.5).sum()),
        "n_negative_above_0.5": int((neg_scores >= 0.5).sum()),
        "co_occurrence_in_positives": {k: v for k, v in co.items() if v},
        "lowest_scoring_positives": [
            {"clip_id": r["clip_id"], "p": round(prob[r["clip_id"]], 4), "labels": r["labels"]}
            for r in worst],
        "highest_scoring_negatives": [
            {"clip_id": r["clip_id"], "p": round(prob[r["clip_id"]], 4), "labels": r["labels"]}
            for r in best_neg],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test-csv", default=None, help="default: from the checkpoint's config")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None, help="default: <run dir>/class_diagnostic.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config", {}))
    test_csv = Path(args.test_csv) if args.test_csv else default_test_csv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] {args.ckpt} test={test_csv} device={device}")

    _scores, _valid, multilogits = score_clips(args.ckpt, test_csv, device)
    rows = [r for r in read_rows(test_csv) if r["clip_id"] in multilogits]

    out: dict = {"checkpoint": str(args.ckpt), "test_csv": str(test_csv), "per_category": {}}
    for category in CATEGORIES:
        out["per_category"][category] = diagnostic(rows, multilogits, category)

    loc = Path(args.ckpt).parent / "per_class_localization.json"
    if loc.exists():
        per_cat = json.loads(loc.read_text(encoding="utf-8")).get("per_category", {})
        for category, rep in per_cat.items():
            if category in out["per_category"]:
                out["per_category"][category]["localization_iou"] = round(rep["mean_best_iou"], 4)

    for category, rep in out["per_category"].items():
        iou = rep.get("localization_iou")
        iou_text = f" tIoU={iou:.3f}" if iou is not None else ""
        print(f"{category:14s} n={rep['n_positives']:3d} "
              f"p(pos)={rep['mean_prob_positive']:.3f} p(neg)={rep['mean_prob_negative']:.3f} "
              f"TP@.5={rep['n_positive_above_0.5']:3d} "
              f"FP@.5={rep['n_negative_above_0.5']:4d}{iou_text}")
        print(f"   co-occur: {rep['co_occurrence_in_positives']}")

    target = Path(args.out) if args.out else Path(args.ckpt).parent / "class_diagnostic.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[diag] written -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
