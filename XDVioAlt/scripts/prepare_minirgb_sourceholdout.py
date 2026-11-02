#!/usr/bin/env python3
"""Build a MiniRGB-aligned source-held-out list for binary head training.

The source-held-out split was first assembled with CLIP visual features on a 64-frame grid. The
live MiniRGB path uses a different feature space and the official 16-frame grid. This script maps
the same XD clip labels onto cached MiniRGB features, extracting only missing clips, and keeps the
official XD test list for final scoring. It writes a new list directory and does not modify source
lists or existing feature files.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safewatch.edge.extractor import FEATURE_DIM
from safewatch.edge.extractor import DEFAULT_INPUT, MiniRGB, frame_grid
from safewatch.edge.video import iter_sample_frames, probe
from scripts.extract_minirgb import VIDEO_DIRS, load_extractor

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


@torch.no_grad()
def extract_clip_batched(model: MiniRGB, video: Path, n_snippets: int) -> np.ndarray:
    """Extract a full MiniRGB sequence without retaining full-clip RGB frame arrays."""
    frame_count = int(probe(video).get("frames") or 0)
    indices = frame_grid(frame_count, n_snippets)
    features = np.empty((len(indices), FEATURE_DIM), dtype=np.float32)
    offset = 0
    for _batch_indices, frames in iter_sample_frames(
            video, indices, size=DEFAULT_INPUT, batch_size=16):
        normalized = (frames - MEAN) / STD
        tensor = torch.from_numpy(normalized).permute(0, 3, 1, 2)
        batch_features = model(tensor).cpu().numpy().astype(np.float32, copy=False)
        end = offset + batch_features.shape[0]
        features[offset:end] = batch_features
        offset = end
    if offset != len(indices):
        raise RuntimeError(f"extracted {offset} features for {len(indices)} requested frames")
    return features


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def find_feature(clip_id: str, roots: list[Path]) -> Path | None:
    for root in roots:
        for split in ("train", "val", "test"):
            candidate = root / split / f"{clip_id}.npy"
            if candidate.is_file():
                return candidate
    return None


def feature_length(path: Path) -> int:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM or array.shape[0] < 1:
        raise ValueError(f"invalid MiniRGB feature shape for {path}: {array.shape}")
    # Spot-check a few temporal rows; a full finite-value scan would reread several GB of cache.
    sample = array[::max(1, array.shape[0] // 8)]
    if not np.isfinite(sample).all():
        raise ValueError(f"non-finite MiniRGB values in sampled rows of {path}")
    return int(array.shape[0])


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lists", default="data/lists_abuse_clip_sourceholdout_20261008_baseline")
    parser.add_argument("--test-lists", default="data/lists_minirgb")
    parser.add_argument("--feature-roots", nargs="+", default=["data/features/minirgb"])
    parser.add_argument("--weights", default="export/onnx/minirgb_w24_distilled.pt")
    parser.add_argument("--out-lists", default="data/lists_binary_minirgb_sourceholdout")
    parser.add_argument("--out-features", default="data/features/minirgb_sourceholdout")
    parser.add_argument("--width", type=int, default=24)
    args = parser.parse_args()

    source = ROOT / args.source_lists
    test_lists = ROOT / args.test_lists
    out_lists, out_features = ROOT / args.out_lists, ROOT / args.out_features
    feature_roots = [ROOT / p for p in args.feature_roots] + [out_features]
    out_features.mkdir(parents=True, exist_ok=True)
    root_video_dirs = {split: ROOT / path for split, path in VIDEO_DIRS.items()}

    # Load the distilled model only if the source split has clips absent from the cache.
    missing: list[tuple[str, dict[str, str], Path]] = []
    prepared: dict[str, list[dict[str, str]]] = {}
    cache_hits = Counter()
    for split in ("train", "val"):
        rows = read_rows(source / f"{split}.csv")
        mapped = []
        for row in rows:
            feat = find_feature(row["clip_id"], feature_roots)
            if feat is not None:
                mapped.append({**row, "T": str(feature_length(feat)), "path": str(feat.relative_to(ROOT))})
                cache_hits[split] += 1
                continue
            video_dir = root_video_dirs["train" if split == "val" else split]
            video = video_dir / f"{row['clip_id']}.mp4"
            if not video.is_file():
                raise FileNotFoundError(f"no cached MiniRGB feature or XD video for {row['clip_id']}")
            target_dir = out_features / split
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{row['clip_id']}.npy"
            mapped.append({**row, "path": str(target.relative_to(ROOT))})
            missing.append((split, row, video))
        prepared[split] = mapped

    if missing:
        torch.set_num_threads(1)
        model = load_extractor(ROOT / args.weights, args.width)
        with torch.no_grad():
            for split, row, video in missing:
                frames = int(probe(video).get("frames") or 0)
                n_snippets = max(1, frames // 16)
                features = extract_clip_batched(model, video, n_snippets)
                target = out_features / split / f"{row['clip_id']}.npy"
                np.save(target, features.astype(np.float32, copy=False))
                new_row = next(r for r in prepared[split] if r["clip_id"] == row["clip_id"])
                new_row["T"] = str(features.shape[0])

    for split, rows in prepared.items():
        write_rows(out_lists / f"{split}.csv", rows)

    # Test features are already produced by the same MiniRGB extractor/grid. Preserve that list.
    test_rows = read_rows(test_lists / "test.csv")
    write_rows(out_lists / "test.csv", test_rows)

    manifest = {
        "purpose": "binary violence head trained with live-compatible MiniRGB and log-mel features",
        "source_lists": str(source.relative_to(ROOT)),
        "test_lists": str(test_lists.relative_to(ROOT)),
        "visual_representation": "MiniRGB distilled from I3D, 1024-d, one frame per 16-frame snippet",
        "cached_features_reused": dict(cache_hits),
        "missing_features_extracted": len(missing),
        "train_abuse_clips": sum("abuse" in r.get("labels", "").split("|") for r in prepared["train"]),
        "validation_abuse_clips": sum("abuse" in r.get("labels", "").split("|") for r in prepared["val"]),
        "test_clips": len(test_rows),
        "test_abuse_clips": sum("abuse" in r.get("labels", "").split("|") for r in test_rows),
        "limitations": [
            "UCF-Crime Abuse augmentation is excluded because its weak abuse category is not individually verified.",
            "The XD validation split contains only 10 abuse clips; it cannot establish abuse reliability.",
            "The official XD test set has been used in prior experiments and is not a pristine untouched test.",
            "A matching feature space enables live inference but does not establish Jetson Nano runtime or accuracy.",
        ],
    }
    (out_lists / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
