#!/usr/bin/env python3
"""Extract live-compatible MiniRGB and log-mel features for locally available UCA videos.

The video is conformed to the live path's canonical 24 fps / mono 16 kHz profile before features
are computed. Results are resumable and stored by UCA split; annotations and labels are untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safewatch.edge.audio import decode_audio, mel_patch_stats, quality_frames  # noqa: E402
from safewatch.edge.normalize import CANONICAL_FPS, normalize_video  # noqa: E402
from safewatch.edge.video import probe  # noqa: E402
from scripts.prepare_minirgb_sourceholdout import extract_clip_batched  # noqa: E402
from scripts.extract_minirgb import load_extractor  # noqa: E402

FEATURE_DIM = 1024
AUDIO_DIM = 132
QUALITY_DIM = 5
STRIDE_FRAMES = 16


def load_video_rows(path: Path) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            previous = unique.get(row["video_name"])
            if previous and previous["split"] != row["split"]:
                raise ValueError(f"video occurs in multiple UCA splits: {row['video_name']}")
            unique[row["video_name"]] = row
    split_order = {"train": 0, "val": 1, "test": 2}
    return sorted(unique.values(), key=lambda row: (split_order.get(row["split"], 9), row["video_name"]))


def valid_feature(path: Path, dims: int, allow_empty: bool = False) -> bool:
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return (arr.ndim == 2 and arr.shape[1] == dims
                and (allow_empty or arr.shape[0] > 0)
                and (arr.shape[0] == 0 or np.isfinite(arr[::max(1, len(arr) // 8)]).all()))
    except Exception:
        return False


def save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".npy", delete=False) as fh:
        temporary = Path(fh.name)
        np.save(fh, values.astype(np.float32, copy=False))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/annotations/uca_intervals_review.csv")
    parser.add_argument("--weights", default="export/onnx/minirgb_w24_distilled.pt")
    parser.add_argument("--video-out", default="data/features/uca_minirgb")
    parser.add_argument("--audio-out", default="data/features/uca_mel")
    parser.add_argument("--quality-out", default="data/features/uca_audio_quality")
    parser.add_argument("--normalized-out", default="data/raw/normalized_uca")
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 processes all matched videos")
    args = parser.parse_args()

    annotation_csv = ROOT / args.annotations
    weight_path = ROOT / args.weights
    video_out, audio_out = ROOT / args.video_out, ROOT / args.audio_out
    quality_out = ROOT / args.quality_out
    normalized_out = ROOT / args.normalized_out
    if not annotation_csv.is_file():
        raise SystemExit(f"annotation table not found: {annotation_csv}")
    if not weight_path.is_file():
        raise SystemExit(f"trained MiniRGB weights not found: {weight_path}")
    rows = load_video_rows(annotation_csv)
    if args.limit:
        rows = rows[:args.limit]

    torch.set_num_threads(1)
    model = load_extractor(weight_path, 24)
    manifest_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    started = time.time()

    for index, row in enumerate(rows, 1):
        source = ROOT / row["video_path"]
        if not source.is_file():
            counts["missing_source"] += 1
            print(f"[uca-features] missing {source}", flush=True)
            continue
        split = row["split"]
        clip_id = Path(row["video_name"]).stem
        normalized_path = normalized_out / split / f"{clip_id}__24fps.mp4"
        if normalized_path.is_file():
            cached_fps = float(probe(normalized_path).get("fps") or 0.0)
        else:
            cached_fps = 0.0
        normalized = (normalized_path if abs(cached_fps - CANONICAL_FPS) <= 0.05 else
                      normalize_video(source, dst=normalized_path, fps=CANONICAL_FPS,
                                      loudnorm=True, force=False))
        info = probe(normalized)
        frame_count = int(info.get("frames") or 0)
        fps = float(info.get("fps") or 0.0)
        if frame_count <= 0 or abs(fps - CANONICAL_FPS) > 0.05:
            counts["normalization_or_probe_error"] += 1
            print(f"[uca-features] skip {clip_id}: canonical probe failed (fps={fps}, frames={frame_count})", flush=True)
            continue

        n_snippets = max(1, frame_count // STRIDE_FRAMES)
        visual_path = video_out / split / f"{clip_id}.npy"
        audio_path = audio_out / split / f"{clip_id}.npy"
        quality_path = quality_out / split / f"{clip_id}.npy"
        visual_ok = valid_feature(visual_path, FEATURE_DIM)
        audio_ok = valid_feature(audio_path, AUDIO_DIM, allow_empty=True)
        quality_ok = valid_feature(quality_path, QUALITY_DIM, allow_empty=True)
        if visual_ok and audio_ok and quality_ok:
            visual = np.load(visual_path, mmap_mode="r", allow_pickle=False)
            audio = np.load(audio_path, mmap_mode="r", allow_pickle=False)
            quality = np.load(quality_path, mmap_mode="r", allow_pickle=False)
            counts["cached"] += 1
        else:
            try:
                visual = (np.load(visual_path, mmap_mode="r", allow_pickle=False) if visual_ok
                          else extract_clip_batched(model, normalized, n_snippets))
                waveform = decode_audio(normalized)
                audio = (np.load(audio_path, mmap_mode="r", allow_pickle=False) if audio_ok
                         else mel_patch_stats(waveform))
                quality = (np.load(quality_path, mmap_mode="r", allow_pickle=False) if quality_ok
                           else quality_frames(waveform))
                if visual.ndim != 2 or visual.shape != (n_snippets, FEATURE_DIM):
                    raise ValueError(f"bad MiniRGB shape {visual.shape}, expected {(n_snippets, FEATURE_DIM)}")
                if audio.ndim != 2 or audio.shape[1] != AUDIO_DIM:
                    raise ValueError(f"bad mel shape {audio.shape}, expected (*, {AUDIO_DIM})")
                if quality.ndim != 2 or quality.shape[1] != QUALITY_DIM:
                    raise ValueError(f"bad audio quality shape {quality.shape}, expected (*, {QUALITY_DIM})")
                if not visual_ok:
                    save_array(visual_path, visual)
                if not audio_ok:
                    save_array(audio_path, audio)
                if not quality_ok:
                    save_array(quality_path, quality)
                counts["extracted"] += 1
            except Exception as exc:
                counts["extraction_error"] += 1
                print(f"[uca-features] skip {clip_id}: {type(exc).__name__}: {exc}", flush=True)
                continue

        manifest_rows.append({
            "clip_id": clip_id,
            "split": split,
            "video_path": row["video_path"],
            "normalized_video": str(normalized.relative_to(ROOT)),
            "visual_path": str(visual_path.relative_to(ROOT)),
            "audio_path": str(audio_path.relative_to(ROOT)),
            "quality_path": str(quality_path.relative_to(ROOT)),
            "n_snippets": int(visual.shape[0]),
            "n_audio_patches": int(audio.shape[0]),
            "fps": fps,
            "stride_frames": STRIDE_FRAMES,
            "snippet_seconds": STRIDE_FRAMES / fps,
        })
        if index % 5 == 0 or index == len(rows):
            print(f"[uca-features] {index}/{len(rows)} | extracted={counts['extracted']} cached={counts['cached']} "
                  f"errors={counts['extraction_error']} | {time.time()-started:.0f}s", flush=True)

    manifest_path = ROOT / "data/features/uca_feature_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "extractor": str(weight_path.relative_to(ROOT)),
        "visual": "trained MiniRGB, 1024-d; one sampled frame per 16 frames at canonical 24 fps",
            "audio": "live log-mel patch statistics, 132-d at 0.96 second hop; aligned later to MiniRGB grid",
            "audio_quality": "five live quality statistics at 0.96 second hop; two visual reliability channels are derived from MiniRGB features",
        "canonical_fps": CANONICAL_FPS,
        "normalization": "same normalize_video profile as raw live inference (H.264, 24 fps, mono 16 kHz, loudnorm)",
        "counts": dict(counts),
        "features_written": len(manifest_rows),
        "videos": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "counts": dict(counts),
                      "features_written": len(manifest_rows), "elapsed_seconds": round(time.time()-started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
