#!/usr/bin/env python
"""Comparison table for the P2/P3 ablation ladder, including the modality-removal robustness rows.

Reads every ``runs/<run>/test_metrics*.json`` written by ``safewatch.eval.runner``:

    test_metrics.json              both modalities
    test_metrics_drop_audio.json   audio removed   (--drop-modality audio)
    test_metrics_drop_visual.json  video removed   (--drop-modality visual)
and prints one row per run with the two AP protocols, the localisation quality and the AP lost when
a modality disappears. That last pair is the number the subject asks for ("rester fonctionnelle
lorsqu'une modalite est degradee ou indisponible") - it must not be a claim.
Usage:
    .venv/bin/python scripts/fusion_table.py            # every run that has test metrics
    .venv/bin/python scripts/fusion_table.py p3full     # only runs whose name contains "p3full"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = ("", "_drop_audio", "_drop_visual")


def load(run_dir: Path, suffix: str) -> dict | None:
    path = run_dir / f"test_metrics{suffix}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ap_of(report: dict | None) -> tuple[float, float] | None:
    if not report:
        return None
    return (report["ap"]["global__pr_auc"], report["ap"]["per_video__average_precision"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", nargs="?", default="", help="substring of the run directory name")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    header = (f"{'run':30s} {'clips':>5s} {'APglob':>7s} {'APclip':>7s} {'IoU':>5s} {'F1@.5':>6s}"
              f" {'-audio':>7s} {'-video':>7s}")
    print(header)
    print("-" * len(header))
    for run_dir in sorted(p for p in Path(args.runs_dir).glob("*_*")
                          if p.is_dir() and args.filter in p.name):
        intact = load(run_dir, "")
        if not intact:
            continue
        intact_ap = ap_of(intact)
        if intact_ap is None:
            continue
        local = intact.get("localization", {})
        drop_audio = ap_of(load(run_dir, "_drop_audio"))
        drop_visual = ap_of(load(run_dir, "_drop_visual"))
        print(f"{run_dir.name:30s} {intact['n_clips']:5d} {intact_ap[0]:7.4f} {intact_ap[1]:7.4f}"
              f" {local.get('mean_best_iou', float('nan')):5.3f}"
              f" {local.get('f1@iou0.5', float('nan')):6.3f}"
              f" {drop_audio[0] if drop_audio else float('nan'):7.4f}"
              f" {drop_visual[0] if drop_visual else float('nan'):7.4f}")
    print("\nlegend: APglob = global pr_auc (Wu-compatible), APclip = mean per-video AP, "
          "IoU = mean best segment IoU, F1@.5 = segment F1 at tIoU 0.5, "
          "-audio/-video = APglob with that modality zeroed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
