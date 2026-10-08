"""Tests for P7 (Edge AI): the lightweight extractor and the seek-free frame sampler.

Two silent failure modes are pinned here, both of which produce *plausible* numbers rather than
errors:

* the frame grid must map snippet ``t`` to frames inside that snippet's window. An off-by-one grid
  leaves the extractor describing video seconds away from the features it claims to replace - the
  same class of bug that once mis-aligned VGGish pooling (``docs/journal.md``, Session 10);
* the sampler must return the frames that were *asked for*. A sampler that silently returns the
  first N frames still yields a full tensor of the right shape, and only a value-carrying test can
  see the difference.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from safewatch.edge.extractor import FEATURE_DIM, MiniRGB, frame_grid

cv2 = pytest.importorskip("cv2")


# --------------------------------------------------------------------------- #
# frame grid
# --------------------------------------------------------------------------- #
def test_frame_grid_covers_the_clip_and_stays_in_range():
    """One index per snippet, monotone, inside the clip - and the last snippet is reached."""
    idx = frame_grid(frame_count=5760, n_snippets=360)
    assert idx.shape == (360,)
    assert idx[0] == 0
    assert np.all(np.diff(idx) > 0)                       # strictly increasing
    assert idx.max() < 5760
    # 5760 frames / 360 snippets = 16 frames per snippet -> the grid must span the clip
    assert idx[-1] >= 5760 - 16


def test_frame_grid_offset_stays_inside_the_snippet_window():
    """``offset`` shifts the sample inside each 16-frame window and never leaves the clip."""
    base = frame_grid(1000, 20, snippet_frames=16, offset=0)
    shifted = frame_grid(1000, 20, snippet_frames=16, offset=8)
    assert np.all(shifted >= base)
    assert np.all(shifted - base <= 15)                   # stays in the same snippet window
    assert shifted.max() < 1000
    with pytest.raises(ValueError):
        frame_grid(100, 0)
    with pytest.raises(ValueError):
        frame_grid(0, 5)


def test_frame_grid_handles_clips_shorter_than_the_snippet_count():
    """Pathological but real: more snippets than frames must still stay in range."""
    idx = frame_grid(frame_count=5, n_snippets=40)
    assert idx.shape == (40,)
    assert idx.min() >= 0 and idx.max() <= 4


# --------------------------------------------------------------------------- #
# extractor
# --------------------------------------------------------------------------- #
def test_minirgb_shapes_match_the_frozen_backbone_contract():
    """``(B, T, 3, H, W) -> (B, T, 1024)``: the shape the head needs, or it cannot be reused."""
    model = MiniRGB(width=16).eval()
    with torch.no_grad():
        many = model(torch.randn(2, 5, 3, 112, 112))
        single = model(torch.randn(2, 3, 112, 112))
    assert many.shape == (2, 5, FEATURE_DIM)
    assert single.shape == (2, FEATURE_DIM)


def test_minirgb_is_lightweight_and_rejects_a_bad_width():
    """The whole point is a small model; a width that is not a multiple of 8 breaks the trunk."""
    light = MiniRGB(width=16)
    assert light.num_params < 300_000
    assert MiniRGB(width=24).num_params > light.num_params
    for bad in (4, 12, 0):
        with pytest.raises(ValueError):
            MiniRGB(width=bad)


def test_minirgb_is_deterministic_in_eval_mode():
    """Two identical forward passes must agree, or the budget numbers are not reproducible."""
    model = MiniRGB(width=16).eval()
    sample = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        first, second = model(sample), model(sample)
    assert torch.equal(first, second)


# --------------------------------------------------------------------------- #
# sampler (needs a real file: the point is that it does not seek)
# --------------------------------------------------------------------------- #
def _write_gradient_video(path, n_frames: int = 40, size: int = 64, fps: float = 12.0) -> None:
    """A video whose frame ``i`` is a constant grey ``10 + 5*i``, so identity is checkable."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available")
    for i in range(n_frames):
        grey = 10 + 5 * i
        writer.write(np.full((size, size, 3), grey, dtype=np.uint8))
    writer.release()


def test_sample_frames_returns_the_frames_that_were_asked_for(tmp_path):
    """A sampler that quietly returns the *first* N frames still produces a full, plausible tensor.

    Only a value-carrying check can tell the difference, which is why the synthetic video encodes
    the frame index in its pixels.
    """
    from safewatch.edge.video import sample_frames

    video = tmp_path / "gradient.avi"
    _write_gradient_video(video, n_frames=40)
    indices = [0, 10, 25, 39]
    frames = sample_frames(video, indices, size=32)

    assert frames.shape == (4, 32, 32, 3)
    assert frames.dtype == np.float32
    assert frames.min() >= 0.0 and frames.max() <= 1.0
    for position, index in enumerate(indices):
        expected = (10 + 5 * index) / 255.0
        # MJPG is lossy, so compare within a tolerance rather than exactly
        assert frames[position].mean() == pytest.approx(expected, abs=0.05), (
            f"position {position} should hold frame {index}")


def test_sample_frames_validates_its_arguments(tmp_path):
    from safewatch.edge.video import probe, sample_frames

    video = tmp_path / "gradient.avi"
    _write_gradient_video(video, n_frames=12)
    info = probe(video)
    assert info["frames"] > 0 and info["width"] == 64
    with pytest.raises(ValueError):
        sample_frames(video, [])
    with pytest.raises(ValueError):
        sample_frames(video, [-1])
    with pytest.raises(OSError):
        sample_frames(tmp_path / "missing.mp4", [0])


# --------------------------------------------------------------------------- #
# audio extraction & quality
# --------------------------------------------------------------------------- #
def test_audio_quality_frames_computes_expected_channels():
    from safewatch.edge.audio import quality_frames

    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)

    stats = quality_frames(sine, sample_rate=sr, hop_s=0.96)
    assert stats.ndim == 2
    assert stats.shape[1] == 5
    assert stats.shape[0] == 2  # 2.0 s / 0.96 s -> 2 patches
    assert np.all(np.isfinite(stats))
    # RMS of 0.5 sine is 0.5 / sqrt(2) ≈ 0.353
    assert stats[0, 0] == pytest.approx(0.353, rel=0.05)


def test_audio_mel_patch_stats_shapes_and_fallbacks():
    from safewatch.edge.audio import mel_patch_stats

    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
    waveform = np.sin(2 * np.pi * 200 * t).astype(np.float32)

    patches = mel_patch_stats(waveform, sample_rate=sr, hop_s=0.96, n_mels=64)
    assert patches.ndim == 2
    assert patches.shape[1] == 132  # 64 means + 64 stds + 4 extras
    assert patches.shape[0] == 2

    # empty waveform fallback
    empty = mel_patch_stats(np.zeros(0, dtype=np.float32))
    assert empty.shape == (0, 132)


def test_detect_acoustic_events_finds_signatures():
    from safewatch.edge.audio import detect_acoustic_events

    sr = 16000
    # Quiet background + sudden high energy burst
    bg = np.zeros(sr, dtype=np.float32)
    burst = 0.9 * np.random.randn(sr).astype(np.float32)
    waveform = np.concatenate([bg, burst])

    events = detect_acoustic_events(waveform, sample_rate=sr)
    assert "impact_transient" in events or "broadband_blast" in events
    assert detect_acoustic_events(np.zeros(0, dtype=np.float32)) == []


# --------------------------------------------------------------------------- #
# ingestion conversion: arbitrary video -> training profile
# --------------------------------------------------------------------------- #
def _normalize_module():
    from safewatch.edge import normalize

    if not normalize.ffmpeg_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return normalize


def test_needs_conversion_flags_an_off_profile_video(tmp_path):
    """A 30 fps MJPG clip is off the training profile (24 fps, H.264) and must be flagged."""
    normalize = _normalize_module()
    video = tmp_path / "thirty.avi"
    _write_gradient_video(video, n_frames=48, fps=30.0)

    info = normalize.probe_streams(video)
    assert info["fps"] == pytest.approx(30.0, abs=0.5)
    assert normalize.needs_conversion(video)


def test_normalize_video_conforms_fps_and_is_idempotent(tmp_path):
    """Transcoding brings the file to the profile; a conformed file is returned untouched."""
    normalize = _normalize_module()
    video = tmp_path / "thirty.avi"
    _write_gradient_video(video, n_frames=48, fps=30.0)

    out = normalize.normalize_video(video, dst=tmp_path / "conformed.mp4")
    assert out == tmp_path / "conformed.mp4" and out.exists()

    info = normalize.probe_streams(out)
    assert info["fps"] == pytest.approx(24.0, abs=0.2)
    assert info["video_codec"] == "h264"
    assert not normalize.needs_conversion(out)          # already conformed -> no more work
    assert normalize.normalize_video(out) == out        # idempotent, no second transcode


# --------------------------------------------------------------------------- #
# MiniRGB feature extraction (scripts/extract_minirgb.py)
# --------------------------------------------------------------------------- #
def _load_extract_minirgb():
    """Load ``scripts/extract_minirgb.py`` (a CLI, not a package module)."""
    import importlib.util
    import sys
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parent.parent / "scripts" / "extract_minirgb.py"
    spec = importlib.util.spec_from_file_location("safewatch_extract_minirgb", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_minirgb_loads_the_saved_weights_not_a_fresh_model(tmp_path):
    """The live-path bug was a randomly-initialised ONNX; the extractor must load its weights.

    A regression here silently reverts the pipeline to scoring noise, so the test proves the loaded
    model reproduces the saved weights *and* differs from a freshly-initialised one.
    """
    module = _load_extract_minirgb()
    from safewatch.edge.extractor import MiniRGB

    model = MiniRGB(feature_dim=1024, width=16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02)                        # deterministic, off from any fresh init
    model.eval()
    weights = tmp_path / "minirgb.pt"
    torch.save({"model": model.state_dict(), "width": 16}, weights)

    loaded = module.load_extractor(weights, 24)
    assert loaded.feature_dim == 1024
    sample = torch.rand(2, 3, 112, 112)
    with torch.no_grad():
        saved_out = model(sample)
        loaded_out = loaded(sample)
        fresh_out = MiniRGB(feature_dim=1024, width=16).eval()(sample)
    assert torch.allclose(loaded_out, saved_out)        # exactly the saved weights
    assert not torch.allclose(loaded_out, fresh_out)    # and NOT a random init


def test_extract_minirgb_returns_one_feature_per_snippet(tmp_path):
    """``extract_clip`` turns a video into ``(T, 1024)`` - the shape the head consumes."""
    module = _load_extract_minirgb()
    from safewatch.edge.extractor import MiniRGB

    video = tmp_path / "clip.avi"
    _write_gradient_video(video, n_frames=48, fps=24.0)
    model = MiniRGB(feature_dim=1024, width=16).eval()

    features = module.extract_clip(model, video, 6)
    assert features.shape == (6, 1024)
    assert np.isfinite(features).all()


