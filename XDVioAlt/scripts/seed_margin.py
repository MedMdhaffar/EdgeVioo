#!/usr/bin/env python
"""Seed margin: how much of a run's score is the seed, and is a comparison bigger than that?

A single-seed difference between two variants is not evidence: the Session 11 gate replication
showed a claimed +0.015 AP effect against a **0.096** spread of the *same* variant across three
seeds. This script answers two questions from the runs already on disk:

* **spread** - for each variant (run tag with its ``_s<seed>`` suffix stripped), the mean and range
  of the test metrics across the seeds available;
* **paired margin** - given two variants (``--margin A B``), the per-seed difference ``A - B`` with
  its mean and range. Pairing matters: seeds trained in the same batch share machine conditions, so
  the paired difference cancels part of the noise.

A margin is only reportable when it is comfortably larger than the seed range of the two variants
compared - the output prints both so the reader can see whether that holds.

Usage:
    .venv/bin/python scripts/seed_margin.py                       # spread of every multi-seed tag
    .venv/bin/python scripts/seed_margin.py --margin p3vggish30ep_cross p2vggish30ep_visual
    # same question, but for the streaming (P4 causal) metrics:
    .venv/bin/python scripts/seed_margin.py --metrics-suffix _causal8 --margin p3vggish30ep_cross \
        p2vggish30ep_visual
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

SEED_SUFFIX = re.compile(r"_s\d+$")


def run_tag(run_dir: Path) -> str:
    """``2026-09-30_p3vggish30ep_cross_s43`` -> ``p3vggish30ep_cross_s43``."""
    return run_dir.name.split("_", 1)[1] if "_" in run_dir.name else run_dir.name


def variant_of(tag: str) -> str:
    return SEED_SUFFIX.sub("", tag)


def seed_of(tag: str) -> int:
    match = SEED_SUFFIX.search(tag)
    return int(match.group()[2:]) if match else 42   # no suffix = the default seed in the configs


def metrics_of(run_dir: Path, suffix: str = "") -> dict | None:
    """Read ``test_metrics<suffix>.json`` (``suffix=""`` = offline, ``"_causal8"`` = streaming)."""
    path = run_dir / f"test_metrics{suffix}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {"ap_global": report["ap"]["global__pr_auc"],
            "ap_per_clip": report["ap"]["per_video__average_precision"],
            "macro_per_class_ap": report["macro_per_class_ap"],
            "mean_best_iou": report["localization"]["mean_best_iou"]}


def collect(runs_dir: Path, suffix: str = "") -> dict[str, dict[int, dict]]:
    """{variant: {seed: metrics}} for every scored run."""
    groups: dict[str, dict[int, dict]] = {}
    for run_dir in sorted(p for p in runs_dir.glob("*_*") if p.is_dir()):
        tag = run_tag(run_dir)
        if "smoke" in tag or "fix_" in tag:
            continue
        metrics = metrics_of(run_dir, suffix)
        if metrics:
            groups.setdefault(variant_of(tag), {})[seed_of(tag)] = metrics
    return groups


def span(values: list[float]) -> str:
    return f"{min(values):.4f}-{max(values):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--metrics-suffix", default="",
                        help="metrics file suffix: '' = offline test_metrics.json, "
                             "'_causal8' = streaming test_metrics_causal8.json")
    parser.add_argument("--margin", nargs=2, metavar=("A", "B"), default=None,
                        help="report the paired per-seed margin A - B")
    args = parser.parse_args()
    groups = collect(Path(args.runs_dir), args.metrics_suffix)

    scope = (f"test_metrics{args.metrics_suffix}.json" if args.metrics_suffix
             else "test_metrics.json")
    print(f"# metrics source: {scope}")
    print(f"{'variant':28s} {'n':>2s} {'APglob mean':>11s} {'APglob range':>15s} "
          f"{'APclip mean':>11s} {'APclip range':>15s}")
    print("-" * 88)
    for variant, by_seed in sorted(groups.items()):
        aps = [m["ap_global"] for m in by_seed.values()]
        clips = [m["ap_per_clip"] for m in by_seed.values()]
        flag = "" if len(aps) > 1 else "   (single seed: no margin known)"
        print(f"{variant:28s} {len(aps):2d} {st.mean(aps):11.4f} {span(aps):>15s} "
              f"{st.mean(clips):11.4f} {span(clips):>15s}{flag}")

    if args.margin:
        a, b = args.margin
        if a not in groups or b not in groups:
            raise SystemExit(f"[seed] unknown variant: {a!r} / {b!r}")
        shared = sorted(set(groups[a]) & set(groups[b]))
        if not shared:
            raise SystemExit(f"[seed] {a} and {b} share no seed")
        print(f"\npaired margin  {a} - {b}  (per seed)")
        print(f"{'seed':>5s} {'A':>8s} {'B':>8s} {'A-B':>8s}")
        diffs = []
        for seed in shared:
            av, bv = groups[a][seed]["ap_global"], groups[b][seed]["ap_global"]
            diffs.append(av - bv)
            print(f"{seed:5d} {av:8.4f} {bv:8.4f} {av - bv:+8.4f}")
        print(f"\nmean margin {st.mean(diffs):+.4f}   range {min(diffs):+.4f} to {max(diffs):+.4f}"
              f"   n={len(diffs)}")
        if len(groups[a]) < 2 or len(groups[b]) < 2:
            print("\nseed spread: unknown - need >=2 seeds per variant; margin not judgeable yet")
            return 0
        spread_a = max(m["ap_global"] for m in groups[a].values()) - min(
            m["ap_global"] for m in groups[a].values())
        spread_b = max(m["ap_global"] for m in groups[b].values()) - min(
            m["ap_global"] for m in groups[b].values())
        bigger = abs(st.mean(diffs)) > max(spread_a, spread_b)
        verdict = 'LARGER' if bigger else 'NOT larger'
        print(f"seed spread: {a} {spread_a:.4f}, {b} {spread_b:.4f} -> margin is {verdict} "
              f"than the wider spread")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
