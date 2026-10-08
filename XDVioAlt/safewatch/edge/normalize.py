"""Conform any input video to the profile the models were trained on (ingestion compatibility).

The benchmark was built from XD-Violence clips at **24 fps** with mono audio, and the snippet time
base is derived from the source fps (``stride_frames / fps``). An arbitrary upload can be 30/29.97
fps, variable-frame-rate, HEVC, 48 kHz stereo, or have a quiet/absent track, so the on-the-fly
extractor would describe a slightly different window than the head was trained on. This module
transcodes such a file to the canonical profile with ffmpeg, so:

* the snippet window is the trained ``16 / 24 = 0.667 s`` regardless of the source fps;
* the decoder never has to guess a container/codec (H.264 ``yuv420p``);
* the audio track is mono 16 kHz, and (optionally) loudness-normalised to a fixed reference so the
  level is comparable to the dataset instead of depending on the recording gain.

Honesty boundary: this fixes **ingestion** compatibility (fps / codec / sample-rate / loudness /
missing track). It does **not** close the *content* domain gap - footage unlike XD-Violence still
scores low, because you cannot transcode a distribution into another one.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

#: Training profile: XD-Violence is 24 fps; audio is decoded to 16 kHz mono everywhere else.
CANONICAL_FPS = 24.0
CANONICAL_SR = 16000
#: Tolerance on the fps comparison (29.97 vs 30.0 must compare equal).
FPS_TOLERANCE = 0.05
#: Default output directory for conformed files (gitignored under data/).
DEFAULT_OUT_DIR = "data/raw/normalized"


def ffmpeg_available() -> bool:
    """Both binaries the conformer needs must be on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_streams(path: str | Path) -> dict:
    """``{fps, has_audio, sample_rate, channels, video_codec}`` via ffprobe (``{}`` on failure)."""
    if not ffmpeg_available():
        return {}
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        streams = json.loads(raw).get("streams", [])
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps = 0.0
    if video is not None:
        num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
        try:
            fps = float(num) / float(den) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0
    return {
        "fps": fps,
        "has_audio": audio is not None,
        "sample_rate": int(audio.get("sample_rate") or 0) if audio else 0,
        "channels": int(audio.get("channels") or 0) if audio else 0,
        "video_codec": (video or {}).get("codec_name", ""),
    }


def needs_conversion(path: str | Path, fps: float = CANONICAL_FPS) -> bool:
    """Whether the file is off-profile in a way that matters (fps, codec, audio channels)."""
    info = probe_streams(path)
    if not info:
        return True                       # unreadable by ffprobe -> try a transcode
    if abs(float(info.get("fps") or 0.0) - fps) > FPS_TOLERANCE:
        return True
    if info.get("video_codec") not in ("h264", ""):
        return True
    if info.get("has_audio") and info.get("channels", 0) != 1:
        return True
    return False


def normalize_video(src: str | Path, dst: str | Path | None = None, fps: float = CANONICAL_FPS,
                    sample_rate: int = CANONICAL_SR, loudnorm: bool = True,
                    force: bool = False) -> Path:
    """Transcode ``src`` to the canonical profile and return the conformed path.

    Returns ``src`` unchanged when it already matches the profile (and ``force`` is false), when
    ffmpeg is unavailable, or when the transcode fails - a conformer that raises would make the demo
    refuse files the pipeline can actually decode, so a failure degrades to the raw input.
    """
    src = Path(src)
    if not ffmpeg_available():
        return src
    if not force and not needs_conversion(src, fps):
        return src

    info = probe_streams(src)
    if dst is None:
        dst = Path(DEFAULT_OUT_DIR) / f"{src.stem}__{fps:g}fps.mp4"
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
           "-map", "0:v:0", "-map", "0:a:0?",          # optional audio (some clips have none)
           "-r", f"{fps:g}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    if info.get("has_audio"):
        cmd += ["-c:a", "aac", "-ac", "1", "-ar", str(sample_rate)]
        if loudnorm:
            # EBU R128 loudness normalisation: makes the audio level comparable to the dataset
            # regardless of the recording gain (clip-normalisation only removes gain/offset).
            cmd += ["-af", "loudnorm=I=-23:TP=-2:LRA=11"]
    cmd.append(str(dst))

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return src
    return dst if dst.exists() else src
