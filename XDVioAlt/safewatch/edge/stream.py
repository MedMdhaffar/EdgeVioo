"""Stream capture: "connecter un flux" for the supervision console.

The subject asks to "import a video **or connect a stream**". The pipeline itself is
seek-based over files, and a live stream has no seek. Rather than build a second,
stateful inference path (which would silently diverge from the tested one), a stream
source is **captured** - N seconds, video *and* audio, time-synchronised by a single
ffmpeg process - into a file, and the existing, tested ``ui.state.analyze_video`` path
runs on that capture. The capture is what the operator reviews, and the audit log keeps
its provenance (``run="stream:<timestamp>"``).

Sources accepted (anything ffmpeg accepts):
    rtsp://host/stream    RTSP camera (``-rtsp_transport tcp`` by default: robust on lossy links)
    /dev/video0           local webcam (V4L2)
    any local file        loopback demo: the file is looped as if it were a live stream

This module is capture-only: it never scores, so it stays free of torch and of the model
artefacts - the console composes it with ``analyze_video``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def probe_source(source: str) -> dict:
    """fps / width / height / audio presence of any ffmpeg source, without decoding a frame.

    Returns an empty dict when the source cannot be probed (the caller then shows a
    friendly error instead of a traceback).
    """
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", source]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    if proc.returncode != 0:
        return {}
    import json

    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    out: dict = {}
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            out["fps"] = _fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
            out["width"] = int(stream.get("width") or 0)
            out["height"] = int(stream.get("height") or 0)
            out["has_video"] = True
        elif stream.get("codec_type") == "audio":
            out["has_audio"] = True
            out["sample_rate"] = int(stream.get("sample_rate") or 0)
    return out


def _fraction(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        num, _, den = value.partition("/")
        den = den or "1"
        return float(num) / float(den) if float(den) else 0.0
    except ValueError:
        return 0.0


def is_file_source(source: str) -> bool:
    return Path(source).is_file()


def capture_stream(source: str, out_path: str | Path, seconds: float,
                   fps: float = 24.0, rtsp_tcp: bool = True) -> Path:
    """Capture ``seconds`` of the source into ``out_path`` (mp4, H.264 + AAC mono 16 kHz).

    The capture profile deliberately matches the training profile of the live/edge
    checkpoints (24 fps, H.264, mono 16 kHz), so the following ``analyze_video`` call
    does not have to re-conform anything. A file source is looped (``-stream_loop -1``)
    so short clips can stand in for a live stream in the demo.

    Raises ``RuntimeError`` with the ffmpeg stderr tail when the source cannot be read.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if is_file_source(source):
        cmd += ["-stream_loop", "-1"]
    if source.startswith("rtsp") and rtsp_tcp:
        cmd += ["-rtsp_transport", "tcp"]
    cmd += ["-i", source, "-t", str(seconds), "-r", str(fps),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "16000", "-ac", "1", "-movflags", "+faststart",
            str(out_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=max(20.0, seconds + 30.0))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"stream capture timed out after {seconds}s: {exc.stderr[-400:]}") from exc
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1024:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
        raise RuntimeError(f"cannot capture from {source!r}" + (f": {tail}" if tail else ""))
    return out_path
