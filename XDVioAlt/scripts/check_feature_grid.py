#!/usr/bin/env python
"""Verify the feature snippet grid: is it frame-based (64 frames) or time-based (~2.67 s)?

The distinction matters: frame annotations in ``test_annotations.txt`` are in frames, so converting
them to snippet indices needs the right rule for *every* clip, including those whose fps is not 24.
This script probes clips whose source is YouTube (id prefix ``v=``, i.e. potentially 30/25 fps) and
compares ``nb_frames / T`` with ``duration / T`` from the feature tensors themselves.

Usage:
    .venv/bin/python scripts/check_feature_grid.py --n 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO = "jherng/xd-violence"
ROOT = Path("data/raw/xd-violence")


def ffprobe_video(path: Path) -> tuple[float, int, float]:
    """(fps, nb_frames, duration_seconds)."""
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
         str(path)], capture_output=True, text=True, check=True).stdout)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, _, den = stream["r_frame_rate"].partition("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return fps, int(stream.get("nb_frames") or 0), float(info["format"]["duration"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="how many clips to probe")
    parser.add_argument("--feature-set", default="i3d_rgb")
    parser.add_argument("--style", default="v=",
                        help="clip-id prefix: 'v=' = YouTube sources, otherwise movie clips")
    args = parser.parse_args()

    api = HfApi()
    files = sorted(os.path.basename(f) for f in api.list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(f"data/{args.feature_set}/test_videos/{args.style}"))
    print(f"{len(files)} test clips with prefix {args.style!r} in {args.feature_set}")
    if not files:
        return 1
    step = max(1, len(files) // args.n)
    picks = files[::step][: args.n]

    print(f"\n{'clip':44s} {'fps':>6s} {'frames':>8s} {'dur(s)':>8s} {'T':>5s} "
          f"{'frames/T':>9s} {'dur/T':>7s}")
    for name in picks:
        stem = name[:-4]
        path = hf_hub_download(REPO, f"data/video/test_videos/{stem}.mp4", repo_type="dataset",
                               local_dir=str(ROOT))
        fps, nbf, dur = ffprobe_video(Path(path))
        feat = ROOT / "data" / args.feature_set / "test_videos" / name
        T = int(np.load(feat, mmap_mode="r").shape[0])
        print(f"{stem[:42]:44s} {fps:6.3f} {nbf:8d} {dur:8.2f} {T:5d} "
              f"{nbf / T:9.2f} {dur / T:7.3f}")

    print("\nInterpretation:")
    print("  frames/T ~ 64 for every fps, dur/T varying with fps  -> FRAME-based grid (64 frames)")
    print("  dur/T ~ 2.67 s for every fps, frames/T varying       -> TIME-based grid (2.67 s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
