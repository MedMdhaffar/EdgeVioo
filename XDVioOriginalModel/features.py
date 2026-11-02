#!/usr/bin/env python3
"""Create temporally aligned RGB-I3D + VGGish features for WSAnodet MIX2."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "video.mp4"
DEFAULT_RGB = ROOT / "video_features/i3d_features/i3d/video_rgb.npy"
DEFAULT_TIMESTAMPS = ROOT / "video_features/i3d_features/i3d/video_timestamps_ms.npy"
DEFAULT_OUTPUT_DIR = ROOT / "aligned_features"
VGGISH_WEIGHTS = (
    ROOT
    / "video_features/models/vggish/hub/checkpoints/vggish-10086976.pth"
)

SAMPLE_RATE = 16_000
STFT_WINDOW_SAMPLES = 400  # 25 ms
STFT_HOP_SAMPLES = 160  # 10 ms
VGGISH_FRAMES = 96
VGGISH_BANDS = 64
VGGISH_DIM = 128
RGB_DIM = 1024
MIX2_DIM = RGB_DIM + VGGISH_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align one VGGish embedding to every RGB-I3D timestamp."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--timestamps", type=Path, default=DEFAULT_TIMESTAMPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def find_ffmpeg() -> Path:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return Path(system_ffmpeg)
    binary_dir = ROOT / ".video_tools/imageio_ffmpeg/binaries"
    vendored_ffmpeg = next(binary_dir.glob("ffmpeg-*"), None)
    if vendored_ffmpeg:
        return vendored_ffmpeg
    raise RuntimeError("FFmpeg is required to decode the audio stream")


def decode_audio(video: Path) -> np.ndarray:
    command = [
        str(find_ffmpeg()),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-c:a",
        "pcm_f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if waveform.size == 0:
        raise RuntimeError("The video contains no decodable audio samples")
    return waveform


def load_mel_features_module():
    path = ROOT / "video_features/models/vggish/vggish_src/mel_features.py"
    spec = importlib.util.spec_from_file_location("xdv_mel_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load VGGish mel frontend from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aligned_log_mel(waveform: np.ndarray, timestamps_ms: np.ndarray) -> np.ndarray:
    """Build one 96x64 example ending at each visual feature timestamp."""
    mel_features = load_mel_features_module()
    support_samples = STFT_WINDOW_SAMPLES + (VGGISH_FRAMES - 1) * STFT_HOP_SAMPLES
    examples = []

    for timestamp_ms in timestamps_ms:
        end = int(round(float(timestamp_ms) * SAMPLE_RATE / 1000.0))
        start = end - support_samples
        segment = np.zeros(support_samples, dtype=np.float32)
        source_start = max(0, start)
        source_end = min(waveform.size, end)
        if source_end > source_start:
            target_start = source_start - start
            segment[target_start : target_start + source_end - source_start] = waveform[
                source_start:source_end
            ]

        example = mel_features.log_mel_spectrogram(
            segment,
            audio_sample_rate=SAMPLE_RATE,
            log_offset=0.01,
            window_length_secs=0.025,
            hop_length_secs=0.010,
            num_mel_bins=VGGISH_BANDS,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7500.0,
        )
        if example.shape != (VGGISH_FRAMES, VGGISH_BANDS):
            raise RuntimeError(f"Unexpected VGGish frontend shape: {example.shape}")
        examples.append(example)

    return np.asarray(examples, dtype=np.float32)


def make_vggish() -> nn.Module:
    layers: list[nn.Module] = []
    channels = 1
    for width in (64, "M", 128, "M", 256, 256, "M", 512, 512, "M"):
        if width == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend(
                [nn.Conv2d(channels, width, kernel_size=3, padding=1), nn.ReLU(True)]
            )
            channels = width

    class VGGishEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(*layers)
            self.embeddings = nn.Sequential(
                nn.Linear(512 * 4 * 6, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, VGGISH_DIM),
                nn.ReLU(True),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            values = self.features(values)
            values = values.transpose(1, 3).transpose(1, 2).contiguous()
            return self.embeddings(values.view(values.size(0), -1))

    model = VGGishEmbedding()
    state = torch.load(VGGISH_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def embed_audio(examples: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    model = make_vggish()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = torch.from_numpy(examples[start : start + batch_size, None])
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    rgb = np.load(args.rgb).astype(np.float32)
    timestamps_ms = np.load(args.timestamps).astype(np.float64)

    if rgb.ndim != 2 or rgb.shape[1] != RGB_DIM:
        raise ValueError(f"RGB features must have shape [T, {RGB_DIM}], got {rgb.shape}")
    if timestamps_ms.ndim != 1 or len(timestamps_ms) != len(rgb):
        raise ValueError(
            f"Expected one timestamp per RGB feature, got {len(timestamps_ms)} for {len(rgb)}"
        )
    if np.any(np.diff(timestamps_ms) <= 0):
        raise ValueError("Visual timestamps must be strictly increasing")

    waveform = decode_audio(args.video)
    log_mel = aligned_log_mel(waveform, timestamps_ms)
    audio = embed_audio(log_mel, args.batch_size)
    if audio.shape != (len(rgb), VGGISH_DIM):
        raise RuntimeError(f"Aligned audio has the wrong shape: {audio.shape}")

    mix2 = np.concatenate((rgb, audio), axis=1).astype(np.float32, copy=False)
    model_input = mix2[None, :, :]
    if model_input.shape[2] != MIX2_DIM:
        raise RuntimeError(f"MIX2 input has the wrong shape: {model_input.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "video_rgb.npy", rgb)
    np.save(args.output_dir / "video_timestamps_ms.npy", timestamps_ms)
    np.save(args.output_dir / "video_vggish_aligned.npy", audio)
    np.save(args.output_dir / "video_mix2.npy", mix2)
    np.save(args.output_dir / "video_model_input.npy", model_input)

    print(f"RGB:         {rgb.shape} float32")
    print(f"Audio:       {audio.shape} float32 (end-aligned to RGB timestamps)")
    print(f"MIX2:        {mix2.shape} float32")
    print(f"Model input: {model_input.shape} float32")
    print(f"Saved to:    {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
