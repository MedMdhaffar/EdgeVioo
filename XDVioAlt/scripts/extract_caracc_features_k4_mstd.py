#!/usr/bin/env python3
"""Extract multi-frame (k=4, mean+std, 2048-d) MiniRGB visual features and aligned audio for AccidentVideos.
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
    np.save(path, values.astype(np.float32))


def extract_clip(
    video_path: Path,
    extractor,
    mini_net,
    visual_path: Path,
    audio_path: Path,
    quality_path: Path,
    stride: int = 16,
) -> int:
    info = probe(video_path)
    total_frames = int(info["frames"])
    n_snippets = max(1, total_frames // stride)
    indices = multi_frame_grid(total_frames, n_snippets, FRAMES_PER_SNIPPET, snippet_frames=stride)
    flat_indices = indices.reshape(-1)

    frames = sample_frames(video_path, flat_indices, size=DEFAULT_INPUT)
    frames = (frames.astype(np.float32) - MEAN) / STD

    if extractor is not None:
        flat_feats = extractor.run(None, {"frames": np.ascontiguousarray(frames.transpose(0, 3, 1, 2))})[0]
    else:
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)
        with torch.no_grad():
            flat_feats = mini_net(frames_t).cpu().numpy()

    grouped = flat_feats.reshape(n_snippets, FRAMES_PER_SNIPPET, BASE_DIM)
    m = grouped.mean(axis=1)
    s = grouped.std(axis=1)
    vis_feats = np.concatenate([m, s], axis=-1)
    save_array(visual_path, vis_feats)

    # Audio extraction
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
        wav_path = Path(tmp_wav.name)
        ok = decode_audio(video_path, wav_path)
        if ok and wav_path.is_file() and wav_path.stat().st_size > 0:
            audio_feats = mel_patch_stats(wav_path, n_snippets, snippet_frames=stride, fps=float(info.get("fps", 24.0)))
            q_feats = quality_frames(wav_path, n_snippets, snippet_frames=stride, fps=float(info.get("fps", 24.0)))
        else:
            audio_feats = np.zeros((n_snippets, AUDIO_DIM), dtype=np.float32)
            q_feats = np.zeros((n_snippets, QUALITY_DIM), dtype=np.float32)

    save_array(audio_path, audio_feats)
    save_array(quality_path, q_feats)
    return n_snippets


def process_csv(csv_path: Path, out_csv: Path, extractor, mini_net) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    updated = []
    for i, r in enumerate(rows, 1):
        cid = r["clip_id"]
        vpath = ROOT / r["video_path"]
        v_out = ROOT / f"data/features/minirgb_k4_mstd/{r['split']}/{cid}.npy"
        a_out = ROOT / f"data/features/mel/{cid}.npy"
        q_out = ROOT / f"data/features/audio_quality/{cid}.npy"

        if not v_out.is_file():
            extract_clip(vpath, extractor, mini_net, v_out, a_out, q_out)

        row_copy = dict(r)
        row_copy["feature_path"] = str(v_out.relative_to(ROOT))
        updated.append(row_copy)

        if i % 15 == 0 or i == len(rows):
            print(f"[{csv_path.name}] {i}/{len(rows)} clips processed")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(updated[0].keys()))
        writer.writeheader()
        writer.writerows(updated)


def main() -> int:
    extractor, mini_net = load_extractor(ROOT / "export/onnx/minirgb_w24_fp32.onnx", ROOT / "export/onnx/minirgb_w24_distilled.pt")
    lists_in = ROOT / "data/lists_caracc_binary"
    lists_out = ROOT / "data/lists_caracc_binary_k4_mstd"

    print("=== Extracting k4_mstd features for AccidentVideos ===")
    process_csv(lists_in / "train.csv", lists_out / "train.csv", extractor, mini_net)
    process_csv(lists_in / "val.csv", lists_out / "val.csv", extractor, mini_net)
    print("AccidentVideos extraction complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
