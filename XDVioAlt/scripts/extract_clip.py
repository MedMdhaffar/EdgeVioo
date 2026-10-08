#!/usr/bin/env python
"""Extract CLIP visual features on the mirror grid, so the fusion head can be trained on a *modern*
frozen backbone instead of the 2019 I3D/VideoSwin features.

Why this exists (the "step 1" experiment of the CLIP/Roadmap discussion)
-----------------------------------------------------------------------
Our best run (``runs/2026-09-29_p3full_cross``) reaches AP 0.798 on the mirror grid using VideoSwin
RGB features - competent but ~8-10 AP below the current XD-Violence frontier, whose gains come
almost entirely from **CLIP features**. Everything downstream of the features (the reliability-gated
fusion head) is held fixed, so swapping in CLIP features isolates exactly one variable.

It writes ``<out>/<split>/<clip_id>.npy`` of shape ``(T, D)`` where ``T`` is the clip's mirror-grid
snippet count (so the visual stream stays aligned with the audio) and ``D`` is the CLIP projection
width (512 for ViT-B/32), plus a split CSV whose ``path`` column points at those features - the same
contract ``scripts/extract_minirgb.py`` uses, so ``MultimodalClipDataset`` consumes them unchanged.

Preprocessing follows CLIP exactly (resize the short side to 224, centre-crop 224, CLIP mean/std)
rather than a square squeeze, because aspect distortion is the kind of fidelity loss this experiment
is trying to avoid.

Measured on this CPU-only box (11 cores), torch eager, 8 threads:

    ViT-B-32 @ 224, bs=32 : ~35 img/s   -> full grid (~294k frames) ~2.3 h
    ViT-B-16 @ 224, bs=16 : ~7 img/s    -> ~11 h (too slow to be useful here)

Hence ``ViT-B-32-quickgelu`` is the default. ONNX Runtime was benchmarked too and gave *no* speedup
over eager torch for this ViT, so the simple path is kept.

Usage:
    .venv/bin/python scripts/extract_clip.py --splits test --limit 8      # smoke test
    .venv/bin/python scripts/extract_clip.py --splits train val test      # step 1 (~3.5 h measured)

Step 1b - temporal aggregation (the follow-up to step 1's negative result)
--------------------------------------------------------------------------
Step 1 kept **one** frame per snippet, so within-window motion was discarded and the AP loss landed
exactly on the motion classes (``fighting`` -0.18, ``explosion`` -0.10). ``--frames-per-snippet K``
samples K frames *inside* each snippet window (never across a boundary) and pools them:

    --pool mean       -> (T, 512)   the window's average appearance
    --pool mean+std   -> (T, 1024)  average + variation: an explicit motion cue

    .venv/bin/python scripts/extract_clip.py --frames-per-snippet 4 --pool mean+std

Outputs get an automatic suffix (``clip_k4_mstd``), so the step-1 features are never overwritten.
Cost scales with K: decode is unchanged (the extra frames sit inside windows already decoded) but
the CLIP forward pass is Kx.
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

#: CLIP's own normalisation constants (openai ViT-B/*).
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 1, 1, 3)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 1, 1, 3)
CLIP_INPUT = 224
VIDEO_DIRS = {
    "train": "data/raw/xd-violence/data/video/train_videos",
    "val": "data/raw/xd-violence/data/video/train_videos",
    "test": "data/raw/xd-violence/data/video/test_videos",
}


def build_visual(model_name: str, pretrained: str):
    """Return the frozen CLIP *visual* tower, in eval mode, on CPU."""
    import open_clip  # imported lazily: keeps the rest of the repo importable without open_clip

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    visual = model.visual
    visual.eval()
    for param in visual.parameters():
        param.requires_grad_(False)
    return visual


def _grid_indices(path: Path, n_snippets: int, n_frames: int = 1) -> np.ndarray:
    """Frame indices for one clip: ``(T, k)`` — ``k`` samples spread *inside* each snippet window.

    The mirror grid's stride is ~64 frames (2.67 s), so ``k`` samples sit between ``i*step`` and
    ``(i+1)*step - 1`` and never leak into the neighbouring snippet. That is what restores the
    within-window temporal detail a single frame per snippet throws away (the step-1 finding:
    after the swap to CLIP the loss landed on the motion classes, ``fighting``/``explosion``).
    """
    from safewatch.edge.video import probe

    frame_count = int(probe(path)["frames"])
    step = max(1, frame_count // int(n_snippets))
    span = max(1, min(step, frame_count))
    offsets = np.unique(np.linspace(0, span - 1, min(int(n_frames), span)).round().astype(int))
    grid = np.arange(int(n_snippets), dtype=np.int64)[:, None] * step + offsets[None, :]
    return np.clip(grid, 0, frame_count - 1)


def _preprocess(frame_bgr: np.ndarray, size: int = CLIP_INPUT) -> np.ndarray:
    """One BGR uint8 frame -> ``(size, size, 3)`` RGB uint8, CLIP-style (short side -> size, centre
    crop). ``INTER_AREA`` is the correct kernel for shrinking; the resize geometry matches CLIP's
    ``Resize``+``CenterCrop``."""
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    new_h, new_w = max(size, round(height * scale)), max(size, round(width * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    return resized[top:top + size, left:left + size]


def _iter_frames(path: Path, wanted: list[int], size: int = CLIP_INPUT):
    """Decode ``path`` once (sequential, never seek) and *yield* the asked frames preprocessed.

    Yielding (rather than returning an array) keeps step-1b cheap in memory: with ``k`` frames per
    snippet the kept-frame count grows by ``k``, and buffering them all would cost ``T*k*150 kB``.
    Only one frame is alive at a time; the caller batches. Frames come back in ascending index order
    (the wanted list is sorted by the caller).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    index, position, last, total = 0, 0, wanted[-1], len(wanted)
    while position < total and index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if index == wanted[position]:
            yield _preprocess(frame, size)
            position += 1
        index += 1
    cap.release()


@torch.no_grad()
def _encode(visual, frames: list[np.ndarray]) -> np.ndarray:
    """Batch of preprocessed uint8 frames -> ``(n, D)`` CLIP embeddings (CLIP normalisation)."""
    batch = np.stack(frames).astype(np.float32) / 255.0
    batch = (batch - CLIP_MEAN) / CLIP_STD
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
    return visual(tensor).numpy().astype(np.float32)


def pooled_dim(dim: int, pool: str) -> int:
    """Output width of :func:`extract_clip` for a given CLIP width and pooling rule."""
    if pool == "mean":
        return int(dim)
    if pool == "mean+std":
        return 2 * int(dim)
    raise ValueError(f"unknown pool {pool!r}; expected 'mean' or 'mean+std'")


@torch.no_grad()
def extract_clip(visual, video: Path, n_snippets: int, batch_size: int,
                 size: int = CLIP_INPUT, n_frames: int = 1,
                 pool: str = "mean") -> np.ndarray:
    """``<video>`` -> ``(T, D)`` CLIP features, with ``n_frames`` samples pooled per snippet.

    ``pool="mean"`` averages the k embeddings inside each snippet window (``D = 512``);
    ``pool="mean+std"`` concatenates that mean with the standard deviation (``D = 1024``), so the
    *variation* across the window becomes an explicit motion cue for the head - which is exactly the
    signal a single frame per snippet discarded.

    Frames stream through ``_iter_frames`` and are encoded in ``batch_size`` chunks, so peak memory
    is the ``(T*k, D)`` embedding array (~2 kB/frame), not the decoded pixels.
    """
    grid = _grid_indices(video, n_snippets, n_frames)              # (T, k)
    flat = grid.reshape(-1)
    wanted = sorted({int(i) for i in flat})
    order = {index: position for position, index in enumerate(wanted)}
    dim = int(visual.output_dim)

    embeddings = np.zeros((len(wanted), dim), dtype=np.float32)
    buffer: list[np.ndarray] = []
    filled = 0
    for frame in _iter_frames(video, wanted, size):
        buffer.append(frame)
        if len(buffer) == batch_size:
            embeddings[filled:filled + len(buffer)] = _encode(visual, buffer)
            filled += len(buffer)
            buffer = []
    if buffer:
        embeddings[filled:filled + len(buffer)] = _encode(visual, buffer)

    per_snippet = embeddings[[order[int(i)] for i in flat]].reshape(
        grid.shape[0], grid.shape[1], dim)
    if pool == "mean":
        return per_snippet.mean(axis=1).astype(np.float32)
    return np.concatenate([per_snippet.mean(axis=1), per_snippet.std(axis=1)],
                          axis=1).astype(np.float32)


def _is_valid(path: Path, n_snippets: int, dim: int) -> bool:
    """A complete, correctly shaped feature file? A crash mid-write leaves a truncated ``.npy`` that
    must be re-extracted, not silently fed to training."""
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return array.ndim == 2 and array.shape[0] == int(n_snippets) and array.shape[1] == dim


#: Per-process handle so a Pool worker loads CLIP once and reuses it across clips.
_WORKER: dict = {}


def _init_worker(model: str, pretrained: str, batch_size: int, size: int,
                 n_frames: int, pool: str) -> None:
    """Load CLIP once per worker; pin to one torch thread so N workers do not oversubscribe."""
    torch.set_num_threads(1)
    _WORKER["visual"] = build_visual(model, pretrained)
    _WORKER["cfg"] = (batch_size, size, n_frames, pool)


def _worker_extract(job: tuple) -> tuple:
    """Extract one clip and write its ``.npy``; returns ``(clip_id, ok, error)``.

    The array is written in the worker (not returned) so nothing large crosses the process boundary.
    """
    clip_id, video, n_snippets, target = job
    batch_size, size, n_frames, pool = _WORKER["cfg"]
    try:
        feats = extract_clip(_WORKER["visual"], Path(video), n_snippets, batch_size, size,
                             n_frames, pool)
    except Exception as exc:                              # one bad clip must not kill the pool
        return (clip_id, 0, f"{type(exc).__name__}: {exc}")
    np.save(target, feats)
    return (clip_id, 1, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--lists", default="data/lists")
    parser.add_argument("--out", default=None,
                        help="default: data/features/clip, or .../clip_k<K>[_mstd] for step-1b, "
                             "which never overwrites the step-1 features")
    parser.add_argument("--out-lists", default=None,
                        help="default: data/lists_clip, or .../lists_clip_k<K>[_mstd]")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8,
                        help="torch threads in sequential mode (ignored when --workers > 1)")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes; each pins itself to 1 torch thread. >1 parallelises the "
                             "whole box (CLIP is CPU-bound here).")
    parser.add_argument("--size", type=int, default=CLIP_INPUT)
    parser.add_argument("--frames-per-snippet", type=int, default=1,
                        help="samples per snippet window, pooled (step-1b). 1 = the step-1 setup; "
                             ">1 restores within-window temporal detail (cost scales with it).")
    parser.add_argument("--pool", default="mean", choices=("mean", "mean+std"),
                        help="how the k frames inside a snippet are pooled: 'mean' keeps width D, "
                             "'mean+std' concatenates the temporal variation (width 2D)")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N clips per split (smoke test); 0 = all")
    parser.add_argument("--max-snippets", type=int, default=2000,
                        help="skip clips whose snippet count exceeds this (a multi-hour clip would "
                             "allocate a huge feature file)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    visual = build_visual(args.model, args.pretrained)
    dim = pooled_dim(int(visual.output_dim), args.pool)
    print(f"[clip] extractor <- {args.model}/{args.pretrained} (dim {dim}, {args.size}px, "
          f"{args.frames_per_snippet} frame(s)/snippet, pool={args.pool}) | "
          f"{args.threads} threads, bs {args.batch_size}", flush=True)

    # Variant-suffixed defaults: step-1 wrote data/features/clip (3.5 h); a step-1b run must not
    # silently overwrite it, so anything other than (k=1, mean) lands in its own directory.
    suffix = "" if (args.frames_per_snippet == 1 and args.pool == "mean") else (
        f"_k{args.frames_per_snippet}" + ("_mstd" if args.pool == "mean+std" else ""))
    out_root = Path(args.out or f"data/features/clip{suffix}")
    list_root = Path(args.out_lists or f"data/lists_clip{suffix}")
    list_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_written, total_skipped = 0, 0
    for split in args.splits:
        rows = list(csv.DictReader(open(Path(args.lists) / f"{split}.csv")))
        if args.limit:
            rows = rows[:args.limit]
        video_dir = Path(VIDEO_DIRS[split])
        feat_dir = out_root / split
        feat_dir.mkdir(parents=True, exist_ok=True)

        out_rows, missing, oversize, skipped = [], 0, 0, 0
        jobs = []
        for row in rows:
            video = video_dir / f"{row['clip_id']}.mp4"
            target = feat_dir / f"{row['clip_id']}.npy"
            if not video.exists():
                missing += 1
                continue
            if int(row["T"]) > args.max_snippets:
                oversize += 1
                continue
            out_rows.append({**row, "path": str(target)})
            if _is_valid(target, int(row["T"]), dim):
                skipped += 1
                continue
            jobs.append((row["clip_id"], str(video), int(row["T"]), str(target)))

        written, failed_ids = 0, []
        if args.workers > 1:
            with mp.Pool(args.workers, _init_worker,
                         (args.model, args.pretrained, args.batch_size, args.size,
                          args.frames_per_snippet, args.pool),
                         maxtasksperchild=8) as pool:
                for clip_id, ok, err in pool.imap_unordered(_worker_extract, jobs, chunksize=2):
                    if ok:
                        written += 1
                    else:
                        failed_ids.append(clip_id)
                        print(f"[clip] FAILED {clip_id[:50]}: {err}", flush=True)
                    if (written + len(failed_ids)) % 25 == 0:
                        print(f"[clip] {split}: {written} written, {skipped} cached "
                              f"[{time.time() - started:.0f}s]", flush=True)
        else:
            torch.set_num_threads(args.threads)
            _WORKER["visual"] = visual
            _WORKER["cfg"] = (args.batch_size, args.size, args.frames_per_snippet, args.pool)
            for position, job in enumerate(jobs):
                clip_id, ok, err = _worker_extract(job)
                if ok:
                    written += 1
                else:
                    failed_ids.append(clip_id)
                    print(f"[clip] FAILED {clip_id[:50]}: {err}", flush=True)
                if (position + 1) % 25 == 0:
                    print(f"[clip] {split}: {written} written, {skipped} cached "
                          f"[{time.time() - started:.0f}s]", flush=True)

        if failed_ids:                     # drop failures from the list so training never sees them
            bad = set(failed_ids)
            out_rows = [r for r in out_rows if r["clip_id"] not in bad]

        if out_rows:
            with (list_root / f"{split}.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
                writer.writeheader()
                writer.writerows(out_rows)
        print(f"[clip] {split}: wrote {written} ({skipped} cached, {len(failed_ids)} failed, "
              f"{missing} no-video, {oversize} oversize) -> {feat_dir}  "
              f"[{time.time() - started:.0f}s]", flush=True)
        total_written += written
        total_skipped += skipped
    print(f"[clip] done: {total_written} extracted, {total_skipped} cached, "
          f"lists -> {list_root} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
