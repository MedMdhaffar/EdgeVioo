#!/usr/bin/env python
"""Extract MiniRGB features for every clip that has a video, so the fusion head can train on the
SAME representation the live/edge path produces at inference.

This closes the train-vs-deploy feature-space gap: the head used to be trained on the released I3D
features while inference fed it MiniRGB output (a different, previously random-init extractor).
Training on MiniRGB features makes the deployed path in-distribution by construction.

For each clip it writes ``<out>/<split>/<clip_id>.npy`` of shape ``(T, 1024)`` where ``T`` is the
clip's official snippet count (so the visual stream aligns with the audio and I3D grids), and a
split CSV whose ``path`` column points at those features (``ClipDataset._load`` reads a ``.npy``
path directly, so no 5-crop logic is involved).

Usage:
    .venv/bin/python scripts/extract_minirgb.py --weights export/onnx/minirgb_w24_distilled.pt \\
        --lists data/lists_official --splits train val test --out data/features/minirgb
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.edge.extractor import DEFAULT_INPUT, FEATURE_DIM, MiniRGB, frame_grid  # noqa: E402
from safewatch.edge.video import probe, sample_frames  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
VIDEO_DIRS = {
    "train": "data/raw/xd-violence/data/video/train_videos",
    "val": "data/raw/xd-violence/data/video/train_videos",
    "test": "data/raw/xd-violence/data/video/test_videos",
}

#: Per-worker handle, set by the Pool initializer (shipping a torch model per job is wasteful).
_WORKER: dict = {"model": None}


def load_extractor(weights: str | Path, width: int) -> MiniRGB:
    """Load the *distilled* MiniRGB weights (never a fresh, random model)."""
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = MiniRGB(feature_dim=1024, width=int(ckpt.get("width", width)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def extract_clip(model: MiniRGB, video: Path, n_snippets: int,
                 size: int = DEFAULT_INPUT) -> np.ndarray:
    """``<video>`` -> ``(n_snippets, 1024)`` MiniRGB features, one sampled frame per snippet."""
    info = probe(video)
    indices = frame_grid(int(info["frames"]), int(n_snippets))
    frames = sample_frames(video, indices, size=size)
    normalized = (frames - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(0, 3, 1, 2)
    return model(tensor).numpy().astype(np.float32)


def _init_worker(weights: str, width: int) -> None:
    """Load the extractor ONCE per worker; decode (not the CNN) is what needs parallelising."""
    torch.set_num_threads(1)
    _WORKER["model"] = load_extractor(weights, width)


def _is_valid(path: Path, n_snippets: int) -> bool:
    """Is an existing feature file complete and the right shape?

    A hard shutdown mid-write leaves a truncated ``.npy``; resume must re-extract it rather than
    trust "the file exists" and feed a corrupt feature into training.
    """
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return (array.ndim == 2 and array.shape[0] == int(n_snippets)
            and array.shape[1] == FEATURE_DIM)


def _process(job: tuple[str, int, str]) -> str:
    """One clip -> one ``.npy``. Never raises: a bad clip is reported, not fatal."""
    video_path, n_snippets, target = job
    target_path = Path(target)
    if target_path.exists() and _is_valid(target_path, n_snippets):
        return "cached"
    try:
        features = extract_clip(_WORKER["model"], Path(video_path), n_snippets)
    except Exception as exc:                     # a corrupt/unreadable clip must not kill the run
        return f"skip:{Path(video_path).stem[:40]}:{exc}"
    np.save(target_path, features)
    return "ok"


def _consume(results, split: str, total: int, started: float,
             written: int = 0, skipped: int = 0) -> tuple[int, int]:
    """Drain a result iterator, counting hits and printing progress every 500 clips."""
    for result in results:
        if result in ("ok", "cached"):
            written += 1
        else:
            skipped += 1
            print(f"[minirgb] {result}", flush=True)
        if (written + skipped) % 500 == 0:
            print(f"[minirgb] {split}: {written + skipped}/{total} "
                  f"[{time.time() - started:.0f}s]", flush=True)
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="export/onnx/minirgb_w24_distilled.pt")
    parser.add_argument("--lists", default="data/lists_official")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out", default="data/features/minirgb")
    parser.add_argument("--out-lists", default="data/lists_minirgb")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--threads", type=int, default=1,
                        help="worker processes; 1 = sequential (the Pool leaked memory and "
                             "OOM-killed this 14 GB box). Each worker uses 1 torch thread.")
    parser.add_argument("--max-snippets", type=int, default=2000,
                        help="skip clips whose official snippet count exceeds this: one clip with "
                             "T=16224 (a 3 h video) allocates ~2.4 GB for a single feature file")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"[minirgb] weights not found: {weights} "
                         "(run scripts/edge_distill.py first - it saves them)")
    print(f"[minirgb] extractor <- {weights} (width {args.width}) | "
          f"{args.threads} workers", flush=True)

    out_root, list_root = Path(args.out), Path(args.out_lists)
    list_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_written, total_skipped = 0, 0
    for split in args.splits:
        rows = list(csv.DictReader(open(Path(args.lists) / f"{split}.csv")))
        video_dir = Path(VIDEO_DIRS[split])
        feat_dir = out_root / split
        feat_dir.mkdir(parents=True, exist_ok=True)

        jobs, out_rows, missing, oversize = [], [], 0, 0
        for row in rows:
            video = video_dir / f"{row['clip_id']}.mp4"
            target = feat_dir / f"{row['clip_id']}.npy"
            if not video.exists():
                missing += 1
                continue
            if int(row["T"]) > args.max_snippets:
                oversize += 1                     # a multi-hour clip; extracted length must match T
                continue
            jobs.append((str(video), int(row["T"]), str(target)))
            out_rows.append({**row, "path": str(target)})

        written, skipped = 0, 0
        if args.threads > 1:
            with mp.Pool(args.threads, _init_worker, (str(weights), args.width)) as pool:
                written, skipped = _consume(pool.imap_unordered(_process, jobs, chunksize=8),
                                            split, len(jobs), started)
        else:
            # Sequential: one clip at a time in a single process. The multiprocessing Pool leaked
            # memory on this box (a worker reached 8 GB and the kernel OOM-killed it), and decode is
            # the bottleneck anyway; a single walk is slower but bounded and cannot OOM.
            torch.set_num_threads(1)
            _WORKER["model"] = load_extractor(weights, args.width)
            written, skipped = _consume(map(_process, jobs), split, len(jobs), started)

        if out_rows:
            with (list_root / f"{split}.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
                writer.writeheader()
                writer.writerows(out_rows)
        print(f"[minirgb] {split}: wrote {written} ({missing} no-video, {oversize} oversize, "
              f"{skipped} skipped) -> {feat_dir}  [{time.time() - started:.0f}s]", flush=True)
        total_written += written
        total_skipped += skipped
    print(f"[minirgb] done: {total_written} features, {total_skipped} skipped, "
          f"lists -> {list_root} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())