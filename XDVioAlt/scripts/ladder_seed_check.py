#!/usr/bin/env python
"""Ladder with seed ranges: which fusion numbers are replicated, and which are one lucky seed?

The report's ladder table ranks the fusion variants, but only ``cross`` and the visual baseline were
trained at 3 seeds. The trouble is ``adaptive``: its extra seeds exist, but they were only ever
scored by the **P8 gate replication**, which records the *clean* number as level 0 of an occlusion
sweep instead of writing a ``test_metrics.json``. Reading the ladder off the seed-42 column alone
therefore shows ``adaptive`` at its **best** seed (0.7281) and hides that its 3-seed mean (0.690)
sits *below* the visual baseline (0.729) with a 0.096 spread - i.e. "all fusions beat visual-only"
would be wrong for that one row.

This script collects the clean global AP for every ladder run from whichever file actually holds it
(``test_metrics.json``, else the ``level == 0.0`` entry of a ``robustness_*_blind.json`` sweep),
then prints the seed spread and the margin against the baseline's mean. ``ap_global`` is a
snippet-level PR-AUC, so the segment rule does not enter it and the two sources are comparable.

Usage:
    .venv/bin/python scripts/ladder_seed_check.py
    .venv/bin/python scripts/ladder_seed_check.py --baseline p2vggish30ep_visual
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

LADDER = ("p3vggish30ep_cross", "p3vggish30ep_late", "p3vggish30ep_early",
          "p3vggish30ep_adaptive", "p2vggish30ep_visual", "p2vggish30ep_audio")


def clean_ap(run_dir: Path) -> tuple[float, str] | None:
    """Clean global AP for a run, and which file it came from.

    Prefers the scored report; falls back to level 0 of a robustness sweep (the gate replication
    never wrote ``test_metrics.json`` for its seeds - see the module docstring).
    """
    metrics = run_dir / "test_metrics.json"
    if metrics.exists():
        report = json.loads(metrics.read_text(encoding="utf-8"))
        return report["ap"]["global__pr_auc"], "test_metrics"
    for sweep in sorted(run_dir.glob("robustness_*_blind.json")):
        levels = json.loads(sweep.read_text(encoding="utf-8")).get("levels", [])
        for entry in levels:
            if entry.get("level") == 0.0:
                return entry["ap_global"], "sweep_L0"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--baseline", default="p2vggish30ep_visual")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    buckets: dict[str, list[float]] = {}
    for tag in LADDER:
        run_dirs = sorted(runs_dir.glob(f"*_{tag}")) + sorted(runs_dir.glob(f"*_{tag}_s[0-9]*"))
        for run_dir in run_dirs:
            found = clean_ap(run_dir)
            if found:
                buckets.setdefault(tag, []).append(found[0])
    if args.baseline not in buckets:
        raise SystemExit(f"[ladder] no runs for baseline {args.baseline!r}")
    base = st.mean(buckets[args.baseline])

    header = (f"{'variant':26s} {'n':>2s} {'mean':>7s} {'range':>17s} {'spread':>7s} "
              f"{'vs baseline':>12s}")
    print(header)
    print("-" * len(header))
    for tag in LADDER:
        values = buckets.get(tag, [])
        if not values:
            continue
        spread = max(values) - min(values)
        flag = "" if len(values) > 1 else "  (1 seed: no spread)"
        print(f"{tag:26s} {len(values):2d} {st.mean(values):7.4f} "
              f"{min(values):.4f}-{max(values):.4f} {spread:7.4f} "
              f"{st.mean(values) - base:+12.4f}{flag}")
    print(f"\nbaseline {args.baseline} mean = {base:.4f} "
          f"(n={len(buckets[args.baseline])}); a variant only 'beats' it when the margin is larger "
          f"than the wider of the two spreads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
