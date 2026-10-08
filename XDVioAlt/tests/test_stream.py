"""Tests for the stream capture layer (``safewatch.edge.stream``) and its console wrapper.

Pinned failure modes: a capture that silently drops the audio track (or writes a 0-byte file)
makes the "connect a stream" feature look functional while the incident sheets lose half their
evidence; a source that cannot be read must raise with a *useful* error, not a traceback.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from safewatch.edge import stream as stream_mod

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _sample_video() -> Path | None:
    for root in ("data/raw/xd-violence/data/video", "data/raw"):
        for path in Path(root).rglob("*.mp4"):
            return path
    return None


SAMPLE = _sample_video()
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not available")


@needs_ffmpeg
@pytest.mark.skipif(SAMPLE is None, reason="no sample mp4 on disk")
def test_probe_source_reads_fps_size_and_audio():
    assert SAMPLE is not None
    info = stream_mod.probe_source(str(SAMPLE))
    assert info.get("has_video") and info.get("width", 0) > 0 and info.get("height", 0) > 0
    assert info.get("fps", 0.0) >= 1.0
    assert "has_audio" in info


def test_probe_source_returns_empty_for_unknown_source(tmp_path):
    assert stream_mod.probe_source(str(tmp_path / "nope.mp4")) == {}


@needs_ffmpeg
@pytest.mark.skipif(SAMPLE is None, reason="no sample mp4 on disk")
def test_capture_stream_writes_a_real_file_with_both_tracks(tmp_path):
    import subprocess
    assert SAMPLE is not None

    out = stream_mod.capture_stream(str(SAMPLE), tmp_path / "cap.mp4", seconds=3.0)
    assert out.exists() and out.stat().st_size > 1024
    info = stream_mod.probe_source(str(out))
    assert info.get("has_video")
    # the capture profile is mono 16 kHz - exactly the training profile the head was trained on
    assert info.get("sample_rate") == 16000
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(out)], capture_output=True, text=True).stdout.strip()
    assert 2.0 <= float(duration) <= 3.6


@needs_ffmpeg
def test_capture_stream_raises_with_a_useful_error_on_bad_source(tmp_path):
    with pytest.raises(RuntimeError, match="cannot capture"):
        stream_mod.capture_stream(str(tmp_path / "missing.mp4"), tmp_path / "x.mp4", seconds=1.0)


def test_state_wrapper_names_captures_by_utc_stamp(tmp_path, monkeypatch):
    """The console wrapper must call the capture with ``stream_<utc stamp>.mp4`` under its
    out_dir - the audit log and the operator both reference that name."""
    import safewatch.ui.state as state

    seen: dict = {}

    def fake(source, out_path, seconds, **kwargs):
        seen["out"] = Path(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"\x00" * 2048)
        return Path(out_path)

    monkeypatch.setattr(stream_mod, "capture_stream", fake)
    out = state.capture_stream("rtsp://example", 10.0, out_dir=tmp_path / "caps")
    assert out == seen["out"]
    assert out.name.startswith("stream_") and out.name.endswith(".mp4")
    assert out.parent == tmp_path / "caps"
