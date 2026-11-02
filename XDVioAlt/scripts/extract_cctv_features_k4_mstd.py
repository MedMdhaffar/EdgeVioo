#!/usr/bin/env python3
"""Extract multi-frame (k=4, pool=mean+std, dim=2048) MiniRGB features for RWF-2000 and SCFD CCTV datasets.

This gives CCTV clips motion-awareness:
  n_snippets * 4 frames -> MiniRGB 1024-d -> chunked by 4 -> [mean (1024), std (1024)] = 2048-d.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safewatch.edge.video import probe, sample_frames  # noqa: E402
from safewatch.edge.extractor import DEFAULT_INPUT, MiniRGB, multi_frame_grid  # noqa: E402
from scripts.extract_minirgb import load_extractor, MEAN, STD  # noqa: E402

FEATURE_DIM = 2048
BASE_DIM = 1024
STRIDE_FRAMES = 16
FRAMES_PER_SNIPPET = 4


def save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp_k4_", suffix=".npy", delete=False) as fh:
        temporary = Path(fh.name)
        np.save(fh, values.astype(np.float32, copy=False))
    temporary.replace(path)


def valid_feature(path: Path, dims: int) -> bool:
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return (arr.ndim == 2 and arr.shape[1] == dims and arr.shape[0] > 0
                and np.isfinite(arr[::max(1, len(arr) // 8)]).all())
    except Exception:
        return False


@torch.no_grad()
def extract_visual_k4_mstd(model: MiniRGB, video: Path, n_snippets: int) -> np.ndarray:
    info = probe(video)
    total_frames = int(info["frames"])
    indices = multi_frame_grid(total_frames, n_snippets, FRAMES_PER_SNIPPET)
    frames = sample_frames(video, indices.reshape(-1), size=DEFAULT_INPUT)
    normalized = (frames - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(0, 3, 1, 2)
    feats = model(tensor).cpu().numpy()  # (n_snippets * 4, 1024)
    reshaped = feats.reshape(n_snippets, FRAMES_PER_SNIPPET, BASE_DIM)
    mean = reshaped.mean(axis=1)
    std = reshaped.std(axis=1)
    return np.concatenate([mean, std], axis=-1)  # (n_snippets, 2048)


def extract_for_csv(csv_path: Path, out_csv: Path, model: MiniRGB, video_out: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    updated_rows = []
    for idx, row in enumerate(reader, 1):
        clip_id = row["clip_id"]
        split = row["split"]
        v_path = ROOT / row["video_path"]
        if not v_path.is_file():
            print(f"[cctv-k4] Missing video: {v_path}")
            continue

        vis_dst = video_out / split / f"{clip_id}.npy"
        n_snippets = max(1, int(row.get("T") or 1))

        if not valid_feature(vis_dst, FEATURE_DIM):
            vis_feats = extract_visual_k4_mstd(model, v_path, n_snippets)
            save_array(vis_dst, vis_feats)

        row_copy = dict(row)
        row_copy["path"] = str(vis_dst.relative_to(ROOT))
        updated_rows.append(row_copy)

        if idx % 50 == 0 or idx == len(reader):
            print(f"[{out_csv.parent.name}/{out_csv.name}] {idx}/{len(reader)} clips processed", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(updated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(updated_rows)

    return updated_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="export/onnx/minirgb_w24_distilled.pt")
    parser.add_argument("--rwf-in", default="data/lists_rwf2000_binary")
    parser.add_argument("--scfd-in", default="data/lists_scfd_binary")
    parser.add_argument("--rwf-out", default="data/lists_rwf2000_binary_k4_mstd")
    parser.add_argument("--scfd-out", default="data/lists_scfd_binary_k4_mstd")
    parser.add_argument("--video-out", default="data/features/minirgb_k4_mstd")
    args = parser.parse_args()

    weight_path = ROOT / args.weights
    if not weight_path.is_file():
        raise SystemExit(f"MiniRGB weights not found: {weight_path}")

    torch.set_num_threads(4)
    model = load_extractor(weight_path, 24)

    video_out = ROOT / args.video_out
    rwf_in, rwf_out = ROOT / args.rwf_in, ROOT / args.rwf_out
    scfd_in, scfd_out = ROOT / args.scfd_in, ROOT / args.scfd_out

    # 1. SCFD
    print("=== Extracting k4_mstd (2048-d) features for SCFD ===")
    extract_for_csv(scfd_in / "train.csv", scfd_out / "train.csv", model, video_out)
    extract_for_csv(scfd_in / "val.csv", scfd_out / "val.csv", model, video_out)

    # 2. RWF-2000
    print("=== Extracting k4_mstd (2048-d) features for RWF-2000 ===")
    extract_for_csv(rwf_in / "train.csv", rwf_out / "train.csv", model, video_out)
    extract_for_csv(rwf_in / "val.csv", rwf_out / "val.csv", model, video_out)

    print("CCTV Multi-Frame Feature Extraction Complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

