#!/usr/bin/env python
"""Comparison table for the P8 robustness sweeps (``scripts/robustness_sweep.py`` outputs).

Reads every ``runs/<run>/robustness_*.json`` and prints one row per degradation level, grouped by
(run, degradation), with the AP lost against that sweep's first (clean) level. The first level of a
sweep is assumed to be the clean reference: attenuation 1.0, noise 60 dB, occlusion 0 s,
drop-snippets 0 - the sweep script always records it.

Reminder printed with the table: these are **feature-space** degradations on precomputed features
(see ``safewatch/eval/robustness.py``), not perturbations of the source media.

Usage:
    .venv/bin/python scripts/robustness_table.py                 # every sweep found
    .venv/bin/python scripts/robustness_table.py p3vggish30ep    # filter on the run name
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", nargs="?", default="", help="substring of the run directory name")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    header = (f"{'run':26s} {'degradation':14s} {'stream':6s} {'level':>7s} {'APglob':>7s} "
              f"{'APclip':>7s} {'macro':>7s} {'IoU':>5s} {'\u0394APglob':>9s}")
    print(header)
    print("-" * len(header))
    found = 0
    for run_dir in sorted(p for p in Path(args.runs_dir).glob("*_*")
                          if p.is_dir() and args.filter in p.name):
        for path in sorted(run_dir.glob("robustness_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            levels = data.get("levels", [])
            if not levels:
                continue
            found += 1
            clean = levels[0]["ap_global"]
            for row in levels:
                print(f"{run_dir.name:26s} {data['kind']:14s} {data['stream']:6s} "
                      f"{row['level']:7.3f} {row['ap_global']:7.4f} {row['ap_per_clip']:7.4f} "
                      f"{row['macro_per_class_ap']:7.4f} {row['mean_best_iou']:5.3f} "
                      f"{clean - row['ap_global']:9.4f}")
            print()
    print(f"{found} sweep file(s). legend: level is gain (1=clean) / dB SNR / occlusion seconds / "
          "drop rate | macro = macro per-class AP")
    print("ΔAPglob = drop against that sweep's clean level (first row), so positive = degradation "
          "cost AP")
    print("space: FEATURE-space degradations (precomputed features), not media - see "
          "safewatch/eval/robustness.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
