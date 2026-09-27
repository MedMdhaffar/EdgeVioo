#!/usr/bin/env python3
"""End-to-end five-crop I3D + aligned VGGish inference for WSAnodet MIX2."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
VIDEO_FEATURES_ROOT = ROOT / "video_features"
sys.path.insert(0, str(VIDEO_FEATURES_ROOT))

from models.i3d.i3d_src.i3d_net import I3D 

from features import (  
    aligned_log_mel,
    decode_audio,
    embed_audio,
    find_ffmpeg,
)
from model import Model  


DEFAULT_VIDEO = ROOT / (
    "Abuse_City.of.God.2002___00-37-20_00-38-02_label_B5-0-0.mp4"
)
DEFAULT_OUTPUT_DIR = ROOT / "five_crop_results"
I3D_WEIGHTS = VIDEO_FEATURES_ROOT / "models/i3d/checkpoints/i3d_rgb.pt"
DETECTOR_WEIGHTS = ROOT / "ckpt/wsanodet_mix2.pkl"

FPS = 24
FRAMES_PER_SNIPPET = 16
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
CROP_NAMES = ("center", "top_left", "top_right", "bottom_left", "bottom_right")
RGB_DIM = 1024
AUDIO_DIM = 128
MIX2_DIM = RGB_DIM + AUDIO_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--vggish-batch-size",
        type=int,
        default=16,
        help="number of aligned log-mel examples embedded together",
    )
    parser.add_argument(
        "--reuse-rgb",
        action="store_true",
        help="reuse rgb_0.npy ... rgb_4.npy already present in output-dir",
    )
    return parser.parse_args()


def normalize_to_24fps(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            str(find_ffmpeg()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={FPS}",
            "-an",
            "-c:v",
            "ffv1",
            str(destination),
        ],
        check=True,
    )


def resize_short_side(frame_rgb: np.ndarray) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    scale = RESIZE_SHORT_SIDE / min(height, width)
    new_width = max(RESIZE_SHORT_SIDE, int(round(width * scale)))
    new_height = max(RESIZE_SHORT_SIDE, int(round(height * scale)))
    if (new_width, new_height) == (width, height):
        return frame_rgb
    return cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def five_crops(frame_bgr: np.ndarray) -> np.ndarray:
    """Return five RGB crops in [crop, channel, height, width] order."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = resize_short_side(frame_rgb)
    height, width = frame_rgb.shape[:2]
    max_y = height - CROP_SIZE
    max_x = width - CROP_SIZE
    positions = (
        (max_y // 2, max_x // 2),
        (0, 0),
        (0, max_x),
        (max_y, 0),
        (max_y, max_x),
    )
    crops = [
        frame_rgb[y : y + CROP_SIZE, x : x + CROP_SIZE]
        for y, x in positions
    ]
    return np.transpose(np.stack(crops), (0, 3, 1, 2))


def load_i3d(device: torch.device) -> I3D:
    model = I3D(num_classes=400, modality="rgb")
    state = torch.load(I3D_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def extract_five_crop_i3d(
    normalized_video: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract [5, T, 1024], retaining a one-frame lookahead like make_gt.py."""
    capture = cv2.VideoCapture(str(normalized_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open normalized video: {normalized_video}")

    model = load_i3d(device)
    features: list[list[np.ndarray]] = [[] for _ in CROP_NAMES]
    timestamps_ms: list[float] = []
    pending: list[np.ndarray] = []
    snippets = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pending.append(five_crops(frame))

            # Requiring a lookahead frame implements floor((N - 1) / 16),
            # matching list/make_gt.py's final-frame removal.
            if len(pending) < FRAMES_PER_SNIPPET + 1:
                continue

            clip = np.stack(pending[:FRAMES_PER_SNIPPET], axis=2)
            pending = pending[FRAMES_PER_SNIPPET:]
            clip_tensor = torch.from_numpy(clip).float()
            clip_tensor = clip_tensor.mul_(2.0 / 255.0).sub_(1.0).to(device)

            with torch.inference_mode():
                batch_features = model(clip_tensor, features=True).cpu().numpy()
            if batch_features.shape != (len(CROP_NAMES), RGB_DIM):
                raise RuntimeError(f"Unexpected I3D output shape: {batch_features.shape}")

            for crop_index, crop_features in enumerate(batch_features):
                features[crop_index].append(crop_features.astype(np.float32))
            snippets += 1
            timestamps_ms.append(snippets * FRAMES_PER_SNIPPET / FPS * 1000.0)
            print(f"I3D snippets: {snippets}", end="\r", flush=True)
    finally:
        capture.release()

    print()
    if snippets == 0:
        raise RuntimeError("The video has fewer than 17 decodable frames")

    rgb = np.asarray(features, dtype=np.float32)
    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    if rgb.shape != (len(CROP_NAMES), snippets, RGB_DIM):
        raise RuntimeError(f"Unexpected stacked I3D shape: {rgb.shape}")

    for crop_index in range(len(CROP_NAMES)):
        np.save(output_dir / f"rgb_{crop_index}.npy", rgb[crop_index])
    np.save(output_dir / "timestamps_ms.npy", timestamps)
    return rgb, timestamps


def load_reused_rgb(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.stack(
        [np.load(output_dir / f"rgb_{index}.npy") for index in range(5)]
    ).astype(np.float32)
    timestamps = np.load(output_dir / "timestamps_ms.npy").astype(np.float64)
    if rgb.ndim != 3 or rgb.shape[0] != 5 or rgb.shape[2] != RGB_DIM:
        raise ValueError(f"Reused RGB must have shape [5, T, {RGB_DIM}], got {rgb.shape}")
    if timestamps.shape != (rgb.shape[1],):
        raise ValueError("Reused timestamps do not match the RGB temporal length")
    return rgb, timestamps


def extract_aligned_audio(
    video: Path,
    timestamps_ms: np.ndarray,
    output_dir: Path,
    batch_size: int,
) -> np.ndarray:
    waveform = decode_audio(video)
    examples = aligned_log_mel(waveform, timestamps_ms)
    audio = embed_audio(examples, batch_size)
    expected_shape = (len(timestamps_ms), AUDIO_DIM)
    if audio.shape != expected_shape:
        raise RuntimeError(f"Expected VGGish {expected_shape}, got {audio.shape}")
    np.save(output_dir / "audio.npy", audio)
    return audio


def build_model_input(
    rgb: np.ndarray, audio: np.ndarray, output_dir: Path
) -> np.ndarray:
    repeated_audio = np.broadcast_to(audio[None], (len(CROP_NAMES), *audio.shape))
    model_input = np.concatenate((rgb, repeated_audio), axis=2).astype(
        np.float32, copy=False
    )
    expected_shape = (len(CROP_NAMES), rgb.shape[1], MIX2_DIM)
    if model_input.shape != expected_shape:
        raise RuntimeError(f"Expected MIX2 {expected_shape}, got {model_input.shape}")
    np.save(output_dir / "video_model_input_5crop.npy", model_input)
    return model_input


def run_detector(
    model_input: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    detector = Model(SimpleNamespace(
        feature_size=MIX2_DIM,
        num_classes=1,
        max_seqlen=model_input.shape[1],
    ))
    state = torch.load(DETECTOR_WEIGHTS, map_location="cpu", weights_only=True)
    detector.load_state_dict(
        {name.removeprefix("module."): value for name, value in state.items()}
    )
    detector.to(device).eval()

    values = torch.from_numpy(model_input).to(device)
    with torch.inference_mode():
        offline_logits, online_logits = detector(values, seq_len=None)
        offline = torch.sigmoid(offline_logits).squeeze(-1).mean(dim=0)
        online = torch.sigmoid(online_logits).squeeze(-1).mean(dim=0)

    offline_np = offline.cpu().numpy()
    online_np = online.cpu().numpy()
    np.save(output_dir / "offline_snippet_scores.npy", offline_np)
    np.save(output_dir / "online_snippet_scores.npy", online_np)
    np.save(
        output_dir / "offline_frame_scores.npy",
        np.repeat(offline_np, FRAMES_PER_SNIPPET),
    )
    np.save(
        output_dir / "online_frame_scores.npy",
        np.repeat(online_np, FRAMES_PER_SNIPPET),
    )
    return offline_np, online_np


def write_results(
    output_dir: Path,
    source: Path,
    model_input: np.ndarray,
    offline: np.ndarray,
    online: np.ndarray,
) -> None:
    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["snippet", "start_seconds", "end_seconds", "offline", "online"]
        )
        for index, (offline_score, online_score) in enumerate(zip(offline, online)):
            writer.writerow(
                [
                    index,
                    index * FRAMES_PER_SNIPPET / FPS,
                    (index + 1) * FRAMES_PER_SNIPPET / FPS,
                    float(offline_score),
                    float(online_score),
                ]
            )

    manifest = {
        "source": str(source.resolve()),
        "fps": FPS,
        "frames_per_snippet": FRAMES_PER_SNIPPET,
        "tail_policy": "discard final frame, then discard incomplete snippet",
        "resize_short_side": RESIZE_SHORT_SIDE,
        "crop_size": CROP_SIZE,
        "crop_order": list(CROP_NAMES),
        "rgb_feature_dim": RGB_DIM,
        "audio_feature_dim": AUDIO_DIM,
        "model_input_shape": list(model_input.shape),
        "checkpoint": str(DETECTOR_WEIGHTS.resolve()),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    source = args.video.resolve()
    output_dir = args.output_dir.resolve()
    device = torch.device(args.device)

    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_rgb:
        rgb, timestamps_ms = load_reused_rgb(output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="xdv_5crop_") as temporary:
            normalized_video = Path(temporary) / "video_24fps.mkv"
            print("Normalizing video to 24 FPS...")
            normalize_to_24fps(source, normalized_video)
            print(f"Extracting five-crop I3D features on {device}...")
            rgb, timestamps_ms = extract_five_crop_i3d(
                normalized_video, output_dir, device
            )

    print("Extracting timestamp-aligned VGGish features...")
    audio = extract_aligned_audio(
        source, timestamps_ms, output_dir, args.vggish_batch_size
    )
    model_input = build_model_input(rgb, audio, output_dir)

    print("Running WSAnodet five-crop inference...")
    offline, online = run_detector(model_input, output_dir, device)
    write_results(output_dir, source, model_input, offline, online)

    print(f"RGB features:  {rgb.shape}")
    print(f"Audio features:{audio.shape}")
    print(f"Model input:   {model_input.shape}")
    print(f"Scores:        {offline.shape} offline, {online.shape} online")
    print(f"Saved to:      {output_dir}")


if __name__ == "__main__":
    main()
