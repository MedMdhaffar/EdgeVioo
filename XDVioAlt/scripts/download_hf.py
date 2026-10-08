#!/usr/bin/env python
"""Bootstrap downloader for the XD-Violence HuggingFace mirror.

Source: https://huggingface.co/datasets/jherng/xd-violence  (MIT-tagged mirror
of the ECCV-2020 XD-Violence dataset; please cite Wu et al. 2020 if you use it).

The mirror is large (~150 GB), so this script downloads deliberately:
  * annotations / split lists  -> a few hundred KB
  * I3D RGB test features      -> ~1.4 GB
  * a handful of sample videos -> for ffprobe + audio checks (no 85 GB pull)

Layout after download (LOCAL_DIR defaults to data/raw/xd-violence):
    data/raw/xd-violence/data/test_annotations.txt
    data/raw/xd-violence/data/test_list.txt
    data/raw/xd-violence/data/train_list.txt
    data/raw/xd-violence/data/i3d_rgb/test_videos/*.npy
    data/raw/xd-violence/data/video/test_videos/<id>.mp4

Usage:
    .venv/bin/python scripts/download_hf.py --meta
    .venv/bin/python scripts/download_hf.py --features-test
    .venv/bin/python scripts/download_hf.py --sample-videos 3
    .venv/bin/python scripts/download_hf.py --all --sample-videos 3
    .venv/bin/python scripts/download_hf.py --list-videos --limit 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "jherng/xd-violence"
REPO_TYPE = "dataset"
DEFAULT_LOCAL_DIR = Path("data/raw/xd-violence")

META_PATTERNS = [
    "data/test_annotations.txt",
    "data/test_list.txt",
    "data/train_list.txt",
]
FEATURES_TEST_PATTERN = "data/i3d_rgb/test_videos/*"
VIDEO_PREFIX = "data/video/test_videos/"
# Label codes encoded in the file names (see the dataset builder script).
LABEL_CODES = {
    "A": "normal",
    "B1": "fighting",
    "B2": "shooting",
    "B4": "riot",
    "B5": "abuse",
    "B6": "car_accident",
    "G": "explosion",
}


def label_of(name: str) -> str:
    """Map a test-video file name to its first (dominant) category code."""
    stem = Path(name).stem
    codes = stem.split("_")[-1].split("-")
    return LABEL_CODES.get(codes[0], "unknown")


def list_test_videos() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    return sorted(f for f in files if f.startswith(VIDEO_PREFIX) and f.endswith(".mp4"))


def pick_samples(files: list[str], count: int, seed: int = 0) -> list[str]:
    """Pick `count` videos spread across categories (normal first, then others)."""
    by_label: dict[str, list[str]] = {}
    for f in files:
        by_label.setdefault(label_of(f), []).append(f)

    order = ["normal", "explosion", "car_accident", "fighting", "abuse", "riot", "shooting"]
    chosen: list[str] = []
    for label in order:
        if len(chosen) >= count:
            break
        candidates = by_label.get(label, [])
        if candidates:
            chosen.append(candidates[seed % len(candidates)])
    # fill up from any remaining category if we still need more
    if len(chosen) < count:
        for _label, candidates in by_label.items():
            for f in candidates:
                if len(chosen) >= count:
                    break
                if f not in chosen:
                    chosen.append(f)
    return chosen[:count]


def download(patterns: list[str], local_dir: Path, max_workers: int = 8) -> None:
    from huggingface_hub import snapshot_download

    print(f"[dl] {len(patterns)} pattern(s) -> {local_dir}")
    for p in patterns:
        print(f"     - {p}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        allow_patterns=patterns,
        max_workers=max_workers,
    )
    print(f"[dl] done: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--meta", action="store_true",
                        help="download annotations + train/test split lists")
    parser.add_argument("--features-test", action="store_true",
                        help="download I3D RGB features for the test split (~1.4 GB)")
    parser.add_argument("--sample-videos", type=int, default=0, metavar="N",
                        help="download N sample test videos (one per category)")
    parser.add_argument("--list-videos", action="store_true",
                        help="only list available test videos and exit")
    parser.add_argument("--limit", type=int, default=20,
                        help="max rows printed by --list-videos")
    parser.add_argument("--all", action="store_true",
                        help="meta + test features (NOT the whole video set)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.list_videos:
        files = list_test_videos()
        print(f"{len(files)} test videos available")
        for f in files[: args.limit]:
            print(f"  {label_of(f):13s} {Path(f).name}")
        return 0

    patterns: list[str] = []
    if args.meta or args.all:
        patterns += META_PATTERNS
    if args.features_test or args.all:
        patterns.append(FEATURES_TEST_PATTERN)
    if patterns:
        download(patterns, args.local_dir, args.workers)

    if args.sample_videos:
        files = list_test_videos()
        samples = pick_samples(files, args.sample_videos)
        print("[dl] sample videos chosen (label, name):")
        for f in samples:
            print(f"     - {label_of(f):13s} {Path(f).name}")
        download(samples, args.local_dir, args.workers)

    if not patterns and not args.sample_videos:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
