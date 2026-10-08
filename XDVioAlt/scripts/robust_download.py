#!/usr/bin/env python
"""Robust mirror downloader: per-file retries, skip-on-failure, fully resumable.

The ``hf download`` CLI aborts the entire batch when a single file fails its consistency check
(observed repeatedly on this mirror), losing hours of progress. This script walks the file list
of the repository prefix, downloads file by file, retries transient failures, and records
permanently failing files to a skip log so the pipeline can continue without them.

Usage:
    .venv/bin/python scripts/robust_download.py --include-prefix data/video/test_videos/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="jherng/xd-violence")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--out", default="data/raw/xd-violence")
    parser.add_argument("--include-prefix", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-log", default=None)
    args = parser.parse_args()

    files = sorted(f for f in HfApi().list_repo_files(args.repo, repo_type=args.repo_type)
                   if f.startswith(args.include_prefix))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[dl] nothing to do for prefix {args.include_prefix}")
        return 0
    print(f"[dl] {len(files)} files under {args.include_prefix}", flush=True)

    out_root = Path(args.out)
    skipped: list[dict] = []
    present = 0
    for index, repo_file in enumerate(files, 1):
        target = out_root / repo_file
        if target.exists() and target.stat().st_size > 0:
            present += 1
            continue
        for attempt in range(1, args.attempts + 1):
            try:
                hf_hub_download(args.repo, repo_file, repo_type=args.repo_type,
                                local_dir=args.out)
                present += 1
                break
            except Exception as exc:  # noqa: BLE001 - any failure should not kill the batch
                if attempt == args.attempts:
                    skipped.append({"file": repo_file,
                                    "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
                    print(f"[dl] SKIP {repo_file} ({type(exc).__name__})", flush=True)
                else:
                    time.sleep(2 * attempt)
        if index % 50 == 0:
            print(f"[dl] {index}/{len(files)} processed | {present} present | "
                  f"{len(skipped)} skipped", flush=True)

    print(f"[dl] done: {present}/{len(files)} present, {len(skipped)} skipped", flush=True)
    if args.skip_log and skipped:
        Path(args.skip_log).write_text(json.dumps(skipped, indent=2), encoding="utf-8")
        print(f"[dl] skip log -> {args.skip_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
