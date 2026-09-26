#!/usr/bin/env python3
"""Extract five-crop I3D and aligned VGGish features for WSAnodet MIX2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
VIDEO_FEATURES_ROOT = ROOT / "video_features"
sys.path.insert(0, str(VIDEO_FEATURES_ROOT))

from models.i3d.i3d_src.i3d_net import I3D

from features import aligned_log_mel, decode_audio, embed_audio, find_ffmpeg


DEFAULT_VIDEO = ROOT / (
    "CarAccident_Fast.Furious.2009___00-42-10_00-42-41_label_B6-0-0.mp4"
)
DEFAULT_OUTPUT_DIR = ROOT / "five_crop_features"
I3D_WEIGHTS = VIDEO_FEATURES_ROOT / "models/i3d/checkpoints/i3d_rgb.pt"

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
        help="device used for I3D extraction (default: cuda if available)",
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
        help="reuse rgb_0.npy ... rgb_4.npy and timestamps_ms.npy in output-dir",
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
    frame_rgb = resize_short_side(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
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
    if not I3D_WEIGHTS.is_file():
        raise FileNotFoundError(f"I3D weights do not exist: {I3D_WEIGHTS}")
    model = I3D(num_classes=400, modality="rgb")
    state = torch.load(I3D_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


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

            # This lookahead matches list/make_gt.py: remove the final frame,
            # then discard any incomplete 16-frame snippet.
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
    expected_shape = (len(CROP_NAMES), snippets, RGB_DIM)
    if rgb.shape != expected_shape:
        raise RuntimeError(f"Expected I3D {expected_shape}, got {rgb.shape}")

    for crop_index, crop_features in enumerate(rgb):
        np.save(output_dir / f"rgb_{crop_index}.npy", crop_features)
    np.save(output_dir / "timestamps_ms.npy", timestamps)
    return rgb, timestamps


def load_reused_rgb(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.stack(
        [np.load(output_dir / f"rgb_{index}.npy") for index in range(len(CROP_NAMES))]
    ).astype(np.float32)
    timestamps = np.load(output_dir / "timestamps_ms.npy").astype(np.float64)
    if rgb.ndim != 3 or rgb.shape[0] != len(CROP_NAMES) or rgb.shape[2] != RGB_DIM:
        raise ValueError(
            f"Reused RGB must have shape [{len(CROP_NAMES)}, T, {RGB_DIM}], "
            f"got {rgb.shape}"
        )
    if timestamps.shape != (rgb.shape[1],):
        raise ValueError("Reused timestamps do not match the RGB temporal length")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Reused timestamps must be strictly increasing")
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


def combine_mix2(rgb: np.ndarray, audio: np.ndarray, output_dir: Path) -> np.ndarray:
    repeated_audio = np.broadcast_to(audio[None], (len(CROP_NAMES), *audio.shape))
    mix2 = np.concatenate((rgb, repeated_audio), axis=2).astype(np.float32, copy=False)
    expected_shape = (len(CROP_NAMES), rgb.shape[1], MIX2_DIM)
    if mix2.shape != expected_shape:
        raise RuntimeError(f"Expected MIX2 {expected_shape}, got {mix2.shape}")
    np.save(output_dir / "mix2_5crop.npy", mix2)
    return mix2


def write_manifest(
    output_dir: Path,
    source: Path,
    rgb: np.ndarray,
    audio: np.ndarray,
    mix2: np.ndarray,
) -> None:
    manifest = {
        "source": str(source),
        "fps": FPS,
        "frames_per_snippet": FRAMES_PER_SNIPPET,
        "tail_policy": "discard final frame, then discard incomplete snippet",
        "resize_short_side": RESIZE_SHORT_SIDE,
        "crop_size": CROP_SIZE,
        "crop_order": list(CROP_NAMES),
        "rgb_shape": list(rgb.shape),
        "audio_shape": list(audio.shape),
        "mix2_shape": list(mix2.shape),
        "files": {
            "rgb": [f"rgb_{index}.npy" for index in range(len(CROP_NAMES))],
            "timestamps_ms": "timestamps_ms.npy",
            "audio": "audio.npy",
            "mix2": "mix2_5crop.npy",
        },
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
    if args.vggish_batch_size < 1:
        raise ValueError("--vggish-batch-size must be positive")

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
    mix2 = combine_mix2(rgb, audio, output_dir)
    write_manifest(output_dir, source, rgb, audio, mix2)

    print(f"RGB features:   {rgb.shape} float32")
    print(f"Audio features: {audio.shape} float32")
    print(f"MIX2 features:  {mix2.shape} float32")
    print(f"Saved to:       {output_dir}")


if __name__ == "__main__":
    main()
