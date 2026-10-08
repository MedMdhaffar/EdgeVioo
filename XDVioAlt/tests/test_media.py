"""Tests for the media artefacts that carry *values*: the motion-saliency zones.

Pinned failure mode: a saliency function that returns full-frame boxes (or boxes at random
positions) still writes a JPEG and still looks like "evidence" on the console. Only a test with a
known moving object can see that the box is where the object actually is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from safewatch.explain import media

cv2 = pytest.importorskip("cv2")


def _write_video(path: Path, moving: bool = True) -> None:
    """320x240, 24 fps, 24 frames: a 48x48 white square - stationary centre, or moving left->right
    through the upper-right quadrant region."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (320, 240))
    assert writer.isOpened()
    for frame_index in range(24):
        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        if moving:
            x = 40 + frame_index * 8          # frames 0-23: x goes 40..224
        else:
            x = 136                          # dead centre, no motion
        cv2.rectangle(canvas, (x, 40), (x + 48, 88), (255, 255, 255), -1)
        writer.write(canvas)
    writer.release()


def test_motion_saliency_finds_the_moving_object_where_it_is(tmp_path):
    """At 1.0 s the square is at x ~ 120-184 (frame 24 = t=1.0s: x = 40 + 24*8 = 232? no - the
    window around 1.0 s spans frames ~12-24, i.e. x 136-232): the box must cover the *right half*
    of the frame, never the left, and never the whole frame."""
    video = tmp_path / "moving.mp4"
    _write_video(video, moving=True)
    entries = media.motion_saliency(video, [1.0], tmp_path, prefix="t")
    assert len(entries) == 1 and entries[0]["boxes"], "moving object must produce a motion zone"
    box = entries[0]["boxes"][0]
    assert box["x"] > 0.25            # object is in the right part of the frame at t=1.0 s
    assert box["x"] + box["w"] <= 1.0
    # and it is a zone, not the whole frame: a full-frame box would defeat the purpose
    assert box["w"] * box["h"] < 0.5
    assert box["motion"] >= media.MOTION_FLOOR


def test_motion_saliency_reports_nothing_on_a_stationary_scene(tmp_path):
    """A still scene must NOT produce a box: global-noise boxes are the failure mode this feature
    is meant to avoid, and the card would then be claiming motion that does not exist."""
    video = tmp_path / "still.mp4"
    _write_video(video, moving=False)
    entries = media.motion_saliency(video, [1.0], tmp_path, prefix="s")
    assert entries[0]["boxes"] == []


def test_motion_saliency_degrades_to_empty_on_unreadable_video(tmp_path):
    entries = media.motion_saliency(tmp_path / "nope.mp4", [1.0, 2.0], tmp_path, prefix="x")
    assert [e["boxes"] for e in entries] == [[], []]


def test_motion_saliency_handles_a_timestamp_at_the_clip_edge(tmp_path):
    """t=0 and t=0.95 (last frame) must not underflow/overflow the frame index."""
    video = tmp_path / "edge.mp4"
    _write_video(video, moving=True)
    entries = media.motion_saliency(video, [0.0, 0.95], tmp_path, prefix="e")
    assert all(e["n_frames"] >= 4 for e in entries)
