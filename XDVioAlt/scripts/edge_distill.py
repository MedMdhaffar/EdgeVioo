#!/usr/bin/env python
"""P7 - can a light CNN actually reproduce the frozen I3D features it replaces?

``scripts/edge_budget.py`` prices the extractor but says nothing about whether its output *means*
anything: :class:`~safewatch.edge.extractor.MiniRGB` is randomly initialised there. This script
answers the fidelity question by **distilling** I3D from raw frames and measuring the result against
two controls that make the number interpretable:

1. **random init** - the untrained extractor (what you would ship if you skipped training);
2. **the train-mean feature** - a constant predictor. If the trained net cannot beat a constant
   vector, it has learned nothing about the video;
3. **trained** - the distilled net.

Honesty boundaries, because this experiment sits on a knife edge:

* the only clips with *both* a video on disk and I3D features are the **800 test clips** (the
  official train split has 0 of 3804 videos here), so the extractor is fitted on test-split video.
  Any detection-AP number computed from these features would be contaminated, so this script reports
  **feature fidelity only** and never an AP;
* fidelity is reported on a **held-out subset of clips**, not on the frames it trained on;
* backbones differ in kind: I3D sees 16 frames at 224x224 with a 3D net, the extractor sees 1 frame
  at 112x112. A low score is therefore a *measurement*, not a bug - it says the edge track needs a
  bigger extractor or a head retrained on edge features.

Usage:
    .venv/bin/python scripts/edge_distill.py --clips 80 --snippets 32 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.edge.extractor import FEATURE_DIM, MiniRGB, frame_grid  # noqa: E402
from safewatch.edge.video import probe, sample_frames  # noqa: E402

VIDEO_GLOB = "data/raw/xd-violence/data/video/test_videos/*.mp4"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_i3d(prefix: str) -> np.ndarray | None:
    """Mean-over-5-crops I3D features -> ``(T, 1024)``, or ``None`` when a shard is missing.

    Mirrors ``ClipDataset._load`` (official layout: one file per crop, mean-pooled), because the
    extractor has to match the features the *trained head* was fed, not some other summary of them.
    """
    crops = []
    for c in range(5):
        path = Path(f"{prefix}__{c}.npy")
        if not path.exists():
            return None
        crops.append(np.load(path, mmap_mode="r"))
    if any(arr.ndim != 2 for arr in crops):
        return None
    stacked = np.stack([np.asarray(a, dtype=np.float32) for a in crops], axis=1)  # (T, 5, D)
    return np.ascontiguousarray(stacked.mean(axis=1))


def build_dataset(clips: list[tuple[str, str, str]], samples_per_clip: int, size: int,
                  verbose: bool = True,
                  ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    """Decode each ``(clip_id, video_path, feature_prefix)`` and pair frames with its I3D target.

    Returns ``(frames, targets, kept_clip_ids, counts)`` with ``frames`` of shape
    ``(N*S, 3, size, size)``. Unusable clips are dropped *and counted out loud*, because a silent
    drop is how a distillation set quietly shrinks to the easy examples.

    Memory note: the arrays are **preallocated in one shot** rather than built as a Python list and
    ``torch.cat``-ed at the end. The list+cat form peaks at twice the data (the list *and* the
    concatenated copy), which OOM-killed the first train-split run on this 14 GB machine
    (`Out of memory: Killed process ... anon-rss:5916968kB`). The shape plan comes from mmap-ing the
    feature headers, so it costs no extra decode.
    """
    plan: list[tuple[str, str, str, int]] = []
    dropped = 0
    for clip_id, video, prefix in clips:
        features_path = Path(f"{prefix}__0.npy")
        if not features_path.exists() or not Path(video).exists():
            dropped += 1
            continue
        total = int(np.load(features_path, mmap_mode="r").shape[0])
        if total < 1:
            dropped += 1
            continue
        plan.append((clip_id, video, prefix, total))
    if not plan:
        raise SystemExit("[distill] no usable (video, feature) pairs")

    total_samples = sum(min(samples_per_clip, total) for _, _, _, total in plan)
    frames = torch.empty((total_samples, 3, size, size), dtype=torch.float32)
    targets = torch.empty((total_samples, FEATURE_DIM), dtype=torch.float32)
    kept: list[str] = []
    counts: list[int] = []          # aligned with ``kept`` (a dropped clip contributes nothing)
    print(f"[distill] {len(plan)} usable clips -> <= {total_samples} samples "
          f"(~{frames.numel() * 4 / 2**30:.1f} GB frames)", flush=True)

    cursor = 0
    for position, (clip_id, video, prefix, total) in enumerate(plan):
        pooled = _load_i3d(prefix)
        if pooled is None:
            dropped += 1
            continue
        # The FEATURES define the snippet grid, not the video's frame count: cv2 reports 5761
        # frames for a clip whose official T is 360, and deriving ceil(frames/16)=361 from it
        # dropped every clip of the first run.
        info = probe(video)
        indices = frame_grid(info["frames"], total)       # one frame index per snippet
        # Spread the samples across the clip rather than taking a prefix: a motion feature cannot be
        # matched from the first quarter of the video alone.
        pick = np.linspace(0, total - 1, min(samples_per_clip, total)).round().astype(int)
        rgb = sample_frames(video, indices[pick], size=size)          # (S, size, size, 3)
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)            # (S, 3, size, size)
        count = int(pick.size)
        frames[cursor:cursor + count] = (tensor - MEAN) / STD
        targets[cursor:cursor + count] = torch.from_numpy(pooled[pick])
        cursor += count
        kept.append(clip_id)
        counts.append(count)
        if verbose and position % 10 == 0:
            print(f"[distill] decoded {position}/{len(plan)}", flush=True)
    print(f"[distill] clips kept {len(kept)}, dropped {dropped}")
    return frames[:cursor], targets[:cursor], kept, counts


def cosine_report(model: MiniRGB, frames: torch.Tensor, targets: torch.Tensor,
                  centre: torch.Tensor | None = None, batch: int = 256) -> dict:
    """Cosine similarity to the target: raw **and** mean-centered.

    Raw cosine is dominated by the target's large constant component - predicting the mean I3D
    feature scores ~0.83 without looking at a single frame - so it cannot show whether the extractor
    tracks the video. The centered value subtracts the train-mean target from both sides and is the
    number that means "did it learn input-dependent structure"; a constant predictor scores 0 there
    by construction. Both are reported so the gap is visible instead of flattering.
    """
    model.eval()
    raw_values, centered_values = [], []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch):
            chunk = frames[start:start + batch]
            predicted = model(chunk)
            truth = targets[start:start + batch]
            raw_values.append(F.cosine_similarity(predicted, truth, dim=-1))
            if centre is not None:
                centered_values.append(F.cosine_similarity(predicted - centre, truth - centre,
                                                           dim=-1))
    raw = torch.cat(raw_values)
    report = {"mean_cosine": float(raw.mean()),
              "p10": float(raw.quantile(0.10)), "p90": float(raw.quantile(0.90))}
    if centered_values:
        centered = torch.cat(centered_values)
        report["mean_cosine_centered"] = float(centered.mean())
        report["p10_centered"] = float(centered.quantile(0.10))
    return report


def load_pairs(limit: int, list_csv: str = "data/lists_official/test.csv",
               video_dir: str | None = None) -> list[tuple[str, str, str]]:
    """Clips that have **both** a video on disk and official I3D features.

    ``list_csv`` / ``video_dir`` let a distillation run on the **train** split (whose videos now
    exist on disk) instead of the test split, which fixes the leak where the extractor was fitted on
    test video. The test split stays the default only because it was the only split with videos when
    this script was written.
    """
    import csv

    video_root = Path(video_dir) if video_dir else Path(VIDEO_GLOB).parent
    pairs: list[tuple[str, str, str]] = []
    for row in csv.DictReader(open(list_csv)):
        video = video_root / f"{row['clip_id']}.mp4"
        if video.exists() and Path(f"{row['prefix']}__0.npy").exists():
            pairs.append((row["clip_id"], str(video), row["prefix"]))
        if len(pairs) >= limit:
            break
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=80, help="how many clips to decode")
    parser.add_argument("--snippets", type=int, default=32, help="samples per clip")
    parser.add_argument("--holdout", type=int, default=16, help="clips kept out of training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="export/onnx/edge_distill_report.json")
    parser.add_argument("--list", default="data/lists_official/test.csv",
                        help="split whose (video, I3D) pairs are distilled")
    parser.add_argument("--video-dir", default=None,
                        help="directory holding <clip_id>.mp4 (default: the test video dir)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    pairs = load_pairs(args.clips, args.list, args.video_dir)
    if len(pairs) <= args.holdout:
        raise SystemExit(f"[distill] only {len(pairs)} usable clips, holdout={args.holdout}")
    print(f"[distill] {len(pairs)} (video, I3D) pairs; "
          f"{len(pairs) - args.holdout} train / {args.holdout} holdout")

    start = time.time()
    frames, targets, kept, counts = build_dataset(pairs, args.snippets, 112)
    print(f"[distill] built {frames.shape[0]} samples in {time.time() - start:.0f}s "
          f"({len(kept)} clips, {min(counts)}-{max(counts)} samples each)")

    # Split by CLIP, not by sample: adjacent snippets of one clip are near-duplicates, so a
    # sample-level split would leak the holdout into training and inflate the fidelity.
    hold_clips = max(1, args.holdout)
    boundary = sum(counts[:len(kept) - hold_clips])
    train_x, train_y = frames[:boundary], targets[:boundary]
    hold_x, hold_y = frames[boundary:], targets[boundary:]
    print(f"[distill] train {train_x.shape[0]} samples / holdout {hold_x.shape[0]} "
          f"({len(kept) - hold_clips} vs {hold_clips} clips)")

    model = MiniRGB(width=args.width)
    # The constant predictor IS the train-mean feature, so it scores 0.0 on the centered metric by
    # construction - that is what makes the centered number interpretable.
    centre = train_y.mean(dim=0, keepdim=True)
    model = MiniRGB(width=args.width)
    random_scores = cosine_report(model, hold_x, hold_y, centre=centre)
    constant = centre.expand(hold_y.shape[0], -1)
    constant_cosine = float(F.cosine_similarity(constant, hold_y, dim=-1).mean())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train_x.shape[0])
        total = 0.0
        for start_at in range(0, train_x.shape[0], args.batch):
            index = order[start_at:start_at + args.batch]
            predicted = model(train_x[index])
            # Cosine loss: the head consumes directions, and a scale-adapted feature is a different
            # problem (the benchmark standardises features anyway).
            loss = (1.0 - F.cosine_similarity(predicted, train_y[index], dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * index.numel()
        schedule.step()
        scored = cosine_report(model, hold_x, hold_y, centre=centre)
        history.append({"epoch": epoch, "train_loss": round(total / train_x.shape[0], 4),
                        "holdout_cosine": round(scored["mean_cosine"], 4)})
        if epoch % 2 == 0 or epoch == 1:
            print(f"[distill] epoch {epoch:3d} loss {total / train_x.shape[0]:.4f} "
                  f"holdout cosine {scored['mean_cosine']:+.4f}", flush=True)

    trained = cosine_report(model, hold_x, hold_y, centre=centre)
    centered = trained.get("mean_cosine_centered", float("nan"))

    # Persist the distilled extractor. This was the missing link: nothing ever saved MiniRGB's
    # weights, so the deployed ONNX was exported from a **randomly initialised** MiniRGB
    # (scripts/edge_budget.py builds a fresh model) and the live path knew nothing about the video.
    # Saving here is what lets the ONNX export - and every downstream feature extraction - use a
    # trained extractor instead of noise.
    weights_path = Path(args.out).with_name(f"minirgb_w{args.width}_distilled.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "width": args.width, "epochs": args.epochs,
                "holdout_cosine_centered": centered, "history": history,
                "trained": trained}, weights_path)
    print(f"[distill] extractor weights -> {weights_path}")
    # A constant predictor scores 0.0 centered, so the centered value IS the gain over the best
    # control. Below ~0.05 the extractor's deviations from the mean are essentially unrelated to the
    # target's - i.e. it rescaled the average I3D vector and learned nothing about the video.
    if centered < 0.05:
        verdict = ("no input-dependent structure: the deviations from the train mean do not track "
                   f"the target (centered cosine {centered:+.4f}) while the raw cosine "
                   f"{trained['mean_cosine']:+.4f} is almost exactly what a CONSTANT vector "
                   f"already scores ({constant_cosine:+.4f})")
    else:
        verdict = (f"input-dependent structure present (centered cosine {centered:+.4f} over the "
                   f"constant predictor)")
    report = {
        "device": f"cpu/{args.threads} threads",
        "clips": {"total": len(pairs), "kept": len(kept), "train": len(kept) - hold_clips,
                  "holdout": hold_clips},
        "samples": {"train": int(train_x.shape[0]), "holdout": int(hold_x.shape[0]),
                    "per_clip_min": min(counts), "per_clip_max": max(counts)},
        "extractor": {"width": args.width, "params": model.num_params},
        "controls": {"random_init_cosine": random_scores["mean_cosine"],
                     "random_init_cosine_centered": random_scores.get("mean_cosine_centered"),
                     "constant_train_mean_cosine": constant_cosine,
                     "constant_train_mean_cosine_centered": 0.0},
        "trained": trained,
        "gain_over_best_control_centered": centered,
        "verdict": verdict,
        "history": history,
        "caveat": ("fitted on TEST-split video (it is the only split with videos here), so no "
                   "detection AP is reported from these features; fidelity only"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[distill] holdout cosine - raw is flattered by the target's constant component;")
    print("           centered (mean removed) is the number that shows input-dependent structure:")
    print(f"    {'':16s} {'raw':>9s} {'centered':>9s}")
    print(f"    {'random init':16s} {random_scores['mean_cosine']:+9.4f} "
          f"{random_scores.get('mean_cosine_centered', float('nan')):+9.4f}")
    print(f"    {'constant (mean)':16s} {constant_cosine:+9.4f} {0.0:+9.4f}")
    print(f"    {'trained':16s} {trained['mean_cosine']:+9.4f} {centered:+9.4f}")
    print(f"[distill] verdict: {verdict}")
    print(f"[distill] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

