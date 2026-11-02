#!/usr/bin/env python3
"""Extract multi-frame (k=4, mean+std, 2048-d) MiniRGB visual features and aligned audio for the SmartDashcam dataset.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safewatch.edge.video import probe, sample_frames  # noqa: E402
from safewatch.edge.extractor import DEFAULT_INPUT, MiniRGB, multi_frame_grid  # noqa: E402
from safewatch.edge.audio import decode_audio, mel_patch_stats, quality_frames  # noqa: E402
from scripts.extract_minirgb import load_extractor, MEAN, STD  # noqa: E402

FEATURE_DIM = 2048
BASE_DIM = 1024
AUDIO_DIM = 132
QUALITY_DIM = 5
FRAMES_PER_SNIPPET = 4


def save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp_dash_", suffix=".npy", delete=False) as fh:
        temporary = Path(fh.name)
        np.save(fh, values.astype(np.float32, copy=False))
    temporary.replace(path)


def valid_feature(path: Path, dims: int, allow_empty: bool = False) -> bool:
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return (arr.ndim == 2 and arr.shape[1] == dims
                and (allow_empty or arr.shape[0] > 0)
                and (arr.shape[0] == 0 or np.isfinite(arr[::max(1, len(arr) // 8)]).all()))
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
    feats = model(tensor).cpu().numpy()
    reshaped = feats.reshape(n_snippets, FRAMES_PER_SNIPPET, BASE_DIM)
    mean = reshaped.mean(axis=1)
    std = reshaped.std(axis=1)
    return np.concatenate([mean, std], axis=-1)


def extract_for_csv(csv_path: Path, out_csv: Path, model: MiniRGB, video_out: Path, audio_out: Path, quality_out: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    updated_rows = []
    for idx, row in enumerate(reader, 1):
        clip_id = row["clip_id"]
        split = row["split"]
        v_path = ROOT / row["video_path"]
        if not v_path.is_file():
            print(f"[dashcam-extract] Missing video: {v_path}")
            continue

        vis_dst = video_out / split / f"{clip_id}.npy"
        aud_dst = audio_out / split / f"{clip_id}.npy"
        qua_dst = quality_out / split / f"{clip_id}.npy"
        n_snippets = max(1, int(row.get("T") or 1))

        # 1. Visual k4_mstd
        if not valid_feature(vis_dst, FEATURE_DIM):
            vis_feats = extract_visual_k4_mstd(model, v_path, n_snippets)
            save_array(vis_dst, vis_feats)

        # 2. Audio features
        if not (valid_feature(aud_dst, AUDIO_DIM, allow_empty=True) and valid_feature(qua_dst, QUALITY_DIM, allow_empty=True)):
            try:
                waveform = decode_audio(v_path)
                if waveform is not None and len(waveform) > 0:
                    aud_feats = mel_patch_stats(waveform)
                    qua_feats = quality_frames(waveform)
                else:
                    aud_feats = np.zeros((0, AUDIO_DIM), dtype=np.float32)
                    qua_feats = np.zeros((0, QUALITY_DIM), dtype=np.float32)
            except Exception:
                aud_feats = np.zeros((0, AUDIO_DIM), dtype=np.float32)
                qua_feats = np.zeros((0, QUALITY_DIM), dtype=np.float32)

            save_array(aud_dst, aud_feats)
            save_array(qua_dst, qua_feats)

        row_copy = dict(row)
        row_copy["path"] = str(vis_dst.relative_to(ROOT))
        updated_rows.append(row_copy)

        if idx % 25 == 0 or idx == len(reader):
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
    parser.add_argument("--in-lists", default="data/lists_dashcam_binary")
    parser.add_argument("--out-lists", default="data/lists_dashcam_binary_k4_mstd")
    parser.add_argument("--video-out", default="data/features/minirgb_k4_mstd")
    parser.add_argument("--audio-out", default="data/features/mel")
    parser.add_argument("--quality-out", default="data/features/audio_quality")
    args = parser.parse_args()

    weight_path = ROOT / args.weights
    torch.set_num_threads(4)
    model = load_extractor(weight_path, 24)

    video_out = ROOT / args.video_out
    audio_out = ROOT / args.audio_out
    quality_out = ROOT / args.quality_out
    in_lists = ROOT / args.in_lists
    out_lists = ROOT / args.out_lists

    print("=== Extracting k4_mstd features for SmartDashcam ===")
    extract_for_csv(in_lists / "train.csv", out_lists / "train.csv", model, video_out, audio_out, quality_out)
    extract_for_csv(in_lists / "val.csv", out_lists / "val.csv", model, video_out, audio_out, quality_out)
    print("SmartDashcam extraction complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

