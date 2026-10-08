#!/usr/bin/env python3
"""Audit audio coverage, temporal alignment, silence, and feature numeric health.

Examples::

    .venv/bin/python scripts/audit_audio_features.py \
        --csv data/lists_official/train.csv data/lists_official/val.csv \
              data/lists_official/test.csv \
        --audio-dir data/features/vggish_snippet --audio-grid snippet \
        --stride-frames 16 --expected-dim 128 --out sheets/audio_audit_vggish.json

    .venv/bin/python scripts/audit_audio_features.py \
        --csv data/lists/train.csv data/lists/val.csv data/lists/test.csv \
        --audio-dir data/features/mel --audio-grid patch --stride-frames 64 \
        --expected-dim 132 --out sheets/audio_audit_mel.json

The quality directory defaults to ``data/features/audio_quality``. A clip is flagged as silent
when every saved patch has RMS at or below -75 dB; the extractor's digital-silence floor is -90 dB.
For ``patch`` features, a few rows of tolerance cover final partial windows and snippet rounding.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safewatch.data.multimodal import audit_audio_rows  # noqa: E402


def audit(csv_paths: list[Path], audio_dir: Path, audio_grid: str, stride_frames: float,
          fps: float, expected_dim: int, quality_dir: Path) -> dict:
    rows = []
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["_source_csv"] = str(csv_path)
                rows.append(row)

    report = audit_audio_rows(rows, audio_dir, audio_grid, expected_dim, stride_frames,
                              fps=fps, quality_dir=quality_dir)
    report["csv_files"] = [str(path) for path in csv_paths]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True,
                        help="one or more matching split CSVs")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--audio-grid", choices=("patch", "snippet"), required=True)
    parser.add_argument("--stride-frames", type=float, required=True,
                        help="visual snippet stride used by these lists")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--quality-dir", type=Path, default=Path("data/features/audio_quality"))
    parser.add_argument("--out", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if args.stride_frames <= 0 or args.fps <= 0:
        parser.error("stride and fps must be positive")

    report = audit(args.csv, args.audio_dir, args.audio_grid, args.stride_frames,
                   args.fps, args.expected_dim, args.quality_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("audio_dir", "audio_grid", "counts", "orphan_feature_files",
                       "quality_dir", "time_rows", "issues_truncated")}, indent=2))
    if args.out:
        print(f"Full audit: {args.out}")
    return int(any(report["counts"][key] for key in
                   ("missing", "load_error", "empty", "wrong_ndim", "wrong_dim",
                    "non_finite", "misaligned", "quality_missing", "quality_load_error",
                    "quality_invalid", "silent_audio", "constant_features")))


if __name__ == "__main__":
    raise SystemExit(main())
