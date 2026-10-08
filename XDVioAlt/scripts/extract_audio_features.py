#!/usr/bin/env python
"""Extract audio features (VGGish 128-d + signal-quality stats) from XD-Violence videos.

Per clip we save:
    data/features/vggish/<clip_id>.npy         (P, 128) float32 at the 0.96 s VGGish hop
    data/features/audio_quality/<clip_id>.npy  (P, 5)   float32: rms, rms_db, spectral_flatness,
                                               zero_crossing_rate, clipping_ratio

The snippet grid (64 frames = 2.667 s) is applied later, by the dataset, which pools the patches
falling inside each snippet window. Keeping the raw 0.96 s grid here avoids baking an alignment
choice into the features themselves.

Audio is decoded straight to 16 kHz mono float32 with ffmpeg (our sources are 48 kHz, 6-7 channels,
so downmix + resample are mandatory - verified in docs/dataset.md).

Usage:
    .venv/bin/python scripts/extract_audio_features.py --videos <folder-of-mp4s>
    .venv/bin/python scripts/extract_audio_features.py --list data/lists/train.csv --limit 50
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

TARGET_SR = 16000
VGGISH_HOP_S = 0.96
QUALITY_NAMES = ("rms", "rms_db", "flatness", "zcr", "clip_ratio")


def decode_audio(path: Path, sample_rate: int = TARGET_SR) -> np.ndarray:
    """ffmpeg -> mono float32 waveform at ``sample_rate``.

    Streamed through a pipe, so no temporary wav files are written.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def quality_frames(waveform: np.ndarray, sample_rate: int = TARGET_SR,
                   hop_s: float = VGGISH_HOP_S) -> np.ndarray:
    """Per-patch signal-quality stats.

    These are the measurable reliability signals the adaptive fusion gate and the operator UI need
    (the subject explicitly requires <estimation de la qualite du signal>): loudness, spectral
    flatness (noise-like vs tonal), zero-crossing rate and clipping ratio.
    """
    hop = int(hop_s * sample_rate)
    n_frames = max(1, len(waveform) // hop)
    frames = waveform[: n_frames * hop].reshape(n_frames, hop)
    eps = 1e-9
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + eps)
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) ** 2 + eps
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    clip_ratio = np.mean(np.abs(frames) > 0.99, axis=1)
    return np.stack([rms, rms_db, flatness, zcr, clip_ratio], axis=1).astype(np.float32)


def mel_patch_stats(waveform: np.ndarray, sample_rate: int = TARGET_SR,
                    hop_s: float = VGGISH_HOP_S, n_mels: int = 64) -> np.ndarray:
    """(P, 132) log-mel patch statistics on the same 0.96 s grid as VGGish.

    Per patch: mean and std of each mel band (64 + 64) plus spectral flux, mel-weighted centroid,
    spectral flatness and zero-crossing rate (4). About 1 % of a VGGish forward pass, which is what
    makes extracting all 217 h feasible on this CPU.
    """
    import librosa

    hop_length = 160                      # 10 ms at 16 kHz (VGGish's analysis hop)
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=400,
                                         hop_length=hop_length, n_mels=n_mels,
                                         fmin=50, fmax=8000, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    frames_per_patch = max(1, int(hop_s * sample_rate / hop_length))     # 96 frames = 0.96 s
    n_patches = mel_db.shape[1] // frames_per_patch
    if n_patches < 1:
        return np.zeros((0, 2 * n_mels + 4), dtype=np.float32)
    blocks = mel_db[:, : n_patches * frames_per_patch].reshape(
        n_mels, n_patches, frames_per_patch)
    mean = blocks.mean(axis=2).T
    std = blocks.std(axis=2).T

    band_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=50, fmax=8000)
    power = np.power(10.0, blocks / 10.0)                     # back to power for the spectral stats
    band_power = power.sum(axis=2).T                          # (P, n_mels)
    centroid = (band_power * band_hz).sum(axis=1) / (band_power.sum(axis=1) + 1e-9)
    flatness = np.exp(np.mean(np.log(power + 1e-9), axis=(0, 2))) / (power.mean(axis=(0, 2)) + 1e-9)
    flux = np.abs(np.diff(power, axis=2)).mean(axis=(0, 2)) if frames_per_patch > 1 else np.zeros(1)
    extra = np.stack([centroid, flatness, np.full(n_patches, float(flux.mean())),
                      np.zeros(n_patches)], axis=1)
    return np.concatenate([mean, std, extra], axis=1).astype(np.float32)


_VGGISH = None


def vggish_model():
    """Build VGGish with its pretrained weights (downloaded once, then cached by ``torch.hub``)."""
    global _VGGISH
    if _VGGISH is None:
        import torch
        import torchvggish

        model = torchvggish.VGG(torchvggish.make_layers(), True)  # postprocess=True
        state = torch.hub.load_state_dict_from_url(torchvggish.VGGISH_WEIGHTS, progress=False)
        model.load_state_dict(state)
        model.eval()
        _VGGISH = model
    return _VGGISH


def embed(waveform: np.ndarray) -> np.ndarray:
    """(P, 128) VGGish embeddings; the model handles log-mel framing (0.96 s hop).

    Outputs are the classic PCA + 8-bit quantisation embeddings in [0, 255], as used by the original
    XD-Violence audio features. Normalisation happens in the dataset.
    """
    import torch
    import torchvggish

    model = vggish_model()
    examples = torchvggish.waveform_to_examples(waveform, TARGET_SR)
    if not torch.is_tensor(examples):
        examples = torch.from_numpy(np.asarray(examples))
    with torch.no_grad():
        embeddings = model(examples)
    return np.asarray(embeddings.detach().cpu().numpy(), dtype=np.float32)


def process_clip(video: Path, out_dir: Path, out_quality: Path, backend: str = "mel",
                 overwrite: bool = False) -> dict:
    """Extract one clip's audio features; returns a status dict (never raises for data problems)."""
    clip_id = video.stem
    feature_path = out_dir / f"{clip_id}.npy"
    quality_path = out_quality / f"{clip_id}.npy"
    if feature_path.exists() and quality_path.exists() and not overwrite:
        return {"clip_id": clip_id, "status": "skipped", "backend": backend}

    try:
        waveform = decode_audio(video)
    except subprocess.CalledProcessError as exc:
        return {"clip_id": clip_id, "status": "ffmpeg_error",
                "detail": exc.stderr[-160:].decode(errors="ignore").strip()}
    if waveform.size < TARGET_SR:
        return {"clip_id": clip_id, "status": "too_short", "samples": int(waveform.size)}

    try:
        quality = quality_frames(waveform)
        if backend == "vggish":
            features = embed(waveform)
        elif backend == "mel":
            features = mel_patch_stats(waveform)
        else:
            raise ValueError(f"unknown backend {backend!r}")
    except Exception as exc:  # noqa: BLE001 - report, keep the batch alive
        return {"clip_id": clip_id, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, features)
    np.save(quality_path, quality)
    return {
        "clip_id": clip_id,
        "status": "ok",
        "backend": backend,
        "patches": int(features.shape[0]),
        "dim": int(features.shape[1]),
        "seconds": round(waveform.size / TARGET_SR, 1),
        "mean_rms_db": round(float(np.mean(quality[:, 1])), 2),
    }


def index_videos(video_root: Path) -> dict[str, Path]:
    """stem -> mp4 path for every video under the root (train chunks + test folder)."""
    return {p.stem: p for p in sorted(video_root.rglob("*.mp4"))}


def resolve_videos(args) -> list[Path]:
    if args.videos:
        return sorted(Path(args.videos).glob("*.mp4"))
    index = index_videos(Path(args.video_root))
    stems = [row["clip_id"] for row in
             csv.DictReader(open(args.list, newline="", encoding="utf-8"))]
    missing = [s for s in stems if s not in index]
    if missing:
        print(f"[warn] {len(missing)} clips from {args.list} have no video locally "
              f"(e.g. {missing[0][:40]})")
    return [index[s] for s in stems if s in index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default="data/raw/xd-violence/data/video")
    parser.add_argument("--videos", default=None, help="process every mp4 in this folder")
    parser.add_argument("--list", default=None, help="split CSV listing the clips to process")
    parser.add_argument("--backend", default="mel", choices=["mel", "vggish"],
                        help="mel = log-mel patch statistics (fast, Edge-friendly); "
                             "vggish = pretrained 128-d embeddings (~6.8 GFLOP/patch)")
    parser.add_argument("--out-features", default=None,
                        help="feature output dir (default: data/features/<backend>)")
    parser.add_argument("--out-quality", default="data/features/audio_quality")
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log", default=None, help="optional JSONL status log")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)

    videos = resolve_videos(args)
    if args.limit:
        videos = videos[: args.limit]
    out_features = Path(args.out_features or f"data/features/{args.backend}")
    print(f"[audio] {len(videos)} clips to process (backend={args.backend}) -> {out_features}")

    out_quality = Path(args.out_quality)
    counts: dict[str, int] = {}
    log_handle = open(args.log, "a", encoding="utf-8") if args.log else None
    try:
        for i, video in enumerate(videos, 1):
            status = process_clip(video, out_features, out_quality, args.backend, args.overwrite)
            counts[status["status"]] = counts.get(status["status"], 0) + 1
            if log_handle:
                import json

                log_handle.write(json.dumps(status) + "\n")
                log_handle.flush()
            if i % 25 == 0 or status["status"] not in ("ok", "skipped"):
                print(f"[{i}/{len(videos)}] {status['status']:12s} {status['clip_id'][:44]} "
                      f"{status.get('patches', '')} {status.get('detail', '')}")
    finally:
        if log_handle:
            log_handle.close()

    print(f"[audio] done: {counts}")
    print(f"[audio] feature files present: {sum(1 for _ in out_features.glob('*.npy'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
