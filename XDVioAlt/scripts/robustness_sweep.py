#!/usr/bin/env python
"""P8 - robustness sweep: score one checkpoint across levels of one degradation.

Everything is scored through ``safewatch.eval.runner.score_clips`` with a perturbation hook, so the
degraded run and the clean run share the same code path, the same ground truth and the same metrics
(this project has already been bitten once by a second scorer drifting: Session 10's misaligned
VGGish read-out).

Levels are **wall-clock or semantic**, never snippet counts: occlusion is given in seconds and
converted per grid, attenuation in [0, 1] where 0 is the only informative limit, noise in dB SNR.

**Feature-space caveat.** These degradations act on precomputed features, not on media - see
``safewatch/eval/robustness.py`` for what can and cannot be regenerated (only the log-mel audio
stream can; VGGish and I3D cannot). Quote these numbers as *representation robustness*, not as
"we added noise to the video".

Usage:
    # audio attenuation: clean -> inaudible, on the reference fusion model
    .venv/bin/python scripts/robustness_sweep.py \\
        --ckpt runs/2026-09-29_p3vggish30ep_cross/ckpt_best.pt \\
        --kind attenuation --stream audio --levels 1.0 0.75 0.5 0.25 0.0

    # camera blocked for 5 s / 15 s / 30 s
    .venv/bin/python scripts/robustness_sweep.py \\
        --ckpt runs/2026-09-29_p3vggish30ep_cross/ckpt_best.pt \\
        --kind occlusion --stream visual --levels 5 15 30 --out runs/<run>/robustness_visual.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.eval import metrics as M  # noqa: E402
from safewatch.eval import robustness as RB  # noqa: E402
from safewatch.eval.runner import (  # noqa: E402
    build_gt,
    checkpoint_config,
    compute_metrics,
    default_test_csv,
    load_intervals,
    load_rows,
    resolve_stride,
    score_clips,
)


def sweep_level(ckpt: str, test_csv: Path, device: torch.device, kind: str, level: float,
                stream: str, cfg: dict, rows: list[dict], intervals: dict, stride: float,
                threshold: float, min_length: int, max_gap: int, fps: float, seed: int,
                flag_reliability: bool) -> dict:
    """Score the test split at one degradation level and return AP + localisation."""
    hook = None
    if RB.is_degraded(kind, level):
        hook = RB.build_perturbation(kind, level, stream, cfg, seed=seed,
                                     flag_reliability=flag_reliability)
    scores, _valid, multilogits = score_clips(ckpt, test_csv, device, perturb=hook)
    gt = build_gt(rows, intervals, stride)
    gt_segments = {cid: M.gt_segments_from_intervals(intervals.get(cid, []), stride)
                   for cid in scores}
    report = compute_metrics(scores, gt, gt_segments, multilogits, rows, stride, threshold,
                             min_length, max_gap, fps=fps)
    return {"kind": kind, "stream": stream, "level": level, "applied": hook is not None,
            "ap_global": report["ap"]["global__pr_auc"],
            "ap_per_clip": report["ap"]["per_video__average_precision"],
            "macro_per_class_ap": report["macro_per_class_ap"],
            "mean_best_iou": report["localization"]["mean_best_iou"],
            "f1_at_0.5": report["localization"]["f1@iou0.5"],
            "delay_seconds": report["localization"]["mean_detection_delay_seconds"],
            "per_class_ap": {k: v["ap"] for k, v in report["per_class_clip_level_ap"].items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--kind", required=True, choices=RB.KINDS)
    parser.add_argument("--stream", default="audio", choices=RB.STREAM_NAMES)
    parser.add_argument("--levels", type=float, nargs="+", required=True,
                        help="attenuation: 1.0 (clean) .. 0.0 | noise: dB SNR | "
                             "occlusion: seconds | drop-snippets: rate")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--intervals", default="data/annotations/test_intervals.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-seconds", type=float, default=5.333)
    parser.add_argument("--max-gap-seconds", type=float, default=2.667)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flag-reliability", action="store_true",
                        help="also mark the degraded stream in the reliability vector "
                             "(adaptive gate only)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None, help="default: <run dir>/robustness_<kind>.json")
    parser.add_argument("--figure", default=None,
                        help="optional PNG path for the degradation curve")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = checkpoint_config(args.ckpt)
    test_csv = Path(args.test_csv) if args.test_csv else default_test_csv(cfg)
    stride = resolve_stride(cfg, None)
    min_length = M.snippets_for_seconds(args.min_seconds, stride, args.fps)
    max_gap = M.snippets_for_seconds(args.max_gap_seconds, stride, args.fps, minimum=0)
    rows = load_rows(test_csv)
    intervals = load_intervals(args.intervals)

    print(f"[sweep] {args.ckpt} kind={args.kind} stream={args.stream} stride={stride:g} "
          f"rule=({min_length},{max_gap}) snippets = "
          f"({min_length * stride / args.fps:.2f},{max_gap * stride / args.fps:.2f}) s "
          f"flag_reliability={args.flag_reliability}")
    label = Path(args.ckpt).parent.name

    rows_out = []
    header = (f"{'level':>8s} {'applied':>7s} {'APglob':>7s} {'APclip':>7s} {'macroAP':>7s} "
              f"{'IoU':>5s} {'F1@.5':>6s}")
    print(header)
    print("-" * len(header))
    for level in args.levels:
        report = sweep_level(args.ckpt, test_csv, device, args.kind, level, args.stream, cfg,
                             rows, intervals, stride, args.threshold, min_length, max_gap,
                             args.fps, args.seed, args.flag_reliability)
        rows_out.append(report)
        print(f"{level:8.3f} {str(report['applied']):>7s} {report['ap_global']:7.4f} "
              f"{report['ap_per_clip']:7.4f} {report['macro_per_class_ap']:7.4f} "
              f"{report['mean_best_iou']:5.3f} {report['f1_at_0.5']:6.3f}")

    clean = rows_out[0]["ap_global"]
    out = {"checkpoint": str(args.ckpt), "test_csv": str(test_csv), "stride_frames": stride,
           "segment_rule": {"min_length": min_length, "max_gap": max_gap,
                            "min_seconds": args.min_seconds,
                            "max_gap_seconds": args.max_gap_seconds},
           "stream": args.stream, "seed": args.seed, "flag_reliability": args.flag_reliability,
           "kind": args.kind,
           "space": "feature (precomputed features; see safewatch/eval/robustness.py)",
           "levels": rows_out,
           "ap_global_drop_vs_clean": {str(r["level"]): clean - r["ap_global"] for r in rows_out}}
    target = Path(args.out) if args.out else Path(args.ckpt).parent / f"robustness_{args.kind}.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[sweep] written -> {target}")
    if args.figure:
        write_figure(out, Path(args.figure), label)
    return 0


def write_figure(result: dict, path: Path, label: str) -> None:
    """Plot AP vs degradation level for the report (best-effort: no display, no hard dependency).

    Two curves are drawn because they answer different questions: global AP (do we still raise an
    alarm?) and macro per-class AP (do we still say *what* it is?). They diverge sharply on audio
    degradations, which is the point of the plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                     # pragma: no cover - optional dependency
        print(f"[sweep] no figure ({exc})")
        return
    levels = [row["level"] for row in result["levels"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(levels, [row["ap_global"] for row in result["levels"]], "o-", label="AP global")
    ax.plot(levels, [row["ap_per_clip"] for row in result["levels"]], "s--", label="AP per-clip")
    ax.plot(levels, [row["macro_per_class_ap"] for row in result["levels"]], "^:",
            label="macro per-class AP")
    unit = {"attenuation": "gain level (1 = clean)", "noise": "SNR (dB)",
            "occlusion": f"occlusion of the {result['stream']} stream (s)",
            "drop-snippets": "dropped snippet rate"}[result["kind"]]
    if result["kind"] in ("attenuation", "noise"):
        ax.invert_xaxis()                        # worse to the right on both scales
    ax.set_xlabel(unit)
    ax.set_ylabel("score")
    ax.set_title(f"{result['kind']} on {result['stream']} - {label}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[sweep] figure -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
