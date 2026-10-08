"""Audio extraction & quality estimation for the Edge track and live inference."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

TARGET_SR = 16000
VGGISH_HOP_S = 0.96
QUALITY_NAMES = ("rms", "rms_db", "flatness", "zcr", "clip_ratio")


def decode_audio(path: str | Path, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Extract audio track as mono float32 waveform at ``sample_rate`` via ffmpeg pipe.

    If the video has no audio track or decoding fails, returns an empty array of shape ``(0,)``.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return np.zeros(0, dtype=np.float32)


def quality_frames(waveform: np.ndarray, sample_rate: int = TARGET_SR,
                   hop_s: float = VGGISH_HOP_S) -> np.ndarray:
    """Per-patch signal-quality statistics: (P, 5) float32 array.

    Channels: rms, rms_db, spectral_flatness, zero_crossing_rate, clipping_ratio.
    """
    if waveform.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    hop = int(hop_s * sample_rate)
    n_frames = max(1, len(waveform) // hop)
    padded_len = n_frames * hop
    if len(waveform) < padded_len:
        waveform = np.pad(waveform, (0, padded_len - len(waveform)))
    frames = waveform[:padded_len].reshape(n_frames, hop)

    eps = 1e-9
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + eps)
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))
    window = np.hanning(hop)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 + eps
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    clip_ratio = np.mean(np.abs(frames) > 0.99, axis=1)

    return np.stack([rms, rms_db, flatness, zcr, clip_ratio], axis=1).astype(np.float32)


def mel_patch_stats(waveform: np.ndarray, sample_rate: int = TARGET_SR,
                    hop_s: float = VGGISH_HOP_S, n_mels: int = 64) -> np.ndarray:
    """(P, 132) log-mel patch statistics matching the model input grid."""
    if waveform.size == 0:
        return np.zeros((0, 2 * n_mels + 4), dtype=np.float32)

    import librosa

    hop_length = 160  # 10 ms at 16 kHz
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=400,
                                         hop_length=hop_length, n_mels=n_mels,
                                         fmin=50, fmax=8000, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    frames_per_patch = max(1, int(hop_s * sample_rate / hop_length))
    n_patches = mel_db.shape[1] // frames_per_patch
    if n_patches < 1:
        n_patches = 1
        pad_width = frames_per_patch - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, max(0, pad_width))), mode="edge")

    blocks = mel_db[:, : n_patches * frames_per_patch].reshape(
        n_mels, n_patches, frames_per_patch)
    mean = blocks.mean(axis=2).T
    std = blocks.std(axis=2).T

    band_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=50, fmax=8000)
    power = np.power(10.0, blocks / 10.0)
    band_power = power.sum(axis=2).T
    centroid = (band_power * band_hz).sum(axis=1) / (band_power.sum(axis=1) + 1e-9)
    flatness = np.exp(np.mean(np.log(power + 1e-9), axis=(0, 2))) / (power.mean(axis=(0, 2)) + 1e-9)
    flux = np.abs(np.diff(power, axis=2)).mean(axis=(0, 2)) if frames_per_patch > 1 else np.zeros(1)
    extra = np.stack([centroid, flatness, np.full(n_patches, float(flux.mean())),
                      np.zeros(n_patches)], axis=1)
    return np.concatenate([mean, std, extra], axis=1).astype(np.float32)


def detect_acoustic_events(waveform: np.ndarray, sample_rate: int = TARGET_SR) -> list[str]:
    """Identify physical sound event signatures (impact, screech, explosion, glass/friction)."""
    if waveform.size == 0:
        return []

    q = quality_frames(waveform, sample_rate=sample_rate)
    if q.shape[0] == 0:
        return []

    events = []
    rms_db = q[:, 1]
    flatness = q[:, 2]
    zcr = q[:, 3]

    max_db = float(np.max(rms_db))
    mean_db = float(np.mean(rms_db))
    db_jump = max_db - mean_db

    if db_jump >= 8.0:
        events.append("impact_transient")  # Sudden acoustic shock / collision
    if float(np.max(flatness)) > 0.45:
        events.append("broadband_blast")   # Explosion / metal blast / debris
    if float(np.min(flatness)) < 0.08 and max_db > -35.0:
        events.append("screech_harmonic")  # Tire skid / horn / alarm
    if float(np.max(zcr)) > 0.35:
        events.append("friction_shatter")  # Glass breaking / metal grinding

    return events
