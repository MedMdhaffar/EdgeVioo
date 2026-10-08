"""Frame sampling for the edge track: decode **sequentially**, never seek.

Measured on this machine (``docs/journal.md``, Session 15):

* sequential decode: **1800-2900 fps** (a 4-minute clip in ~3.2 s);
* ``CAP_PROP_POS_FRAMES`` seek: **25-50 ms per frame** (the decoder restarts from the previous
  keyframe), i.e. ~10x the cost of reading the frames it skipped.

So :func:`sample_frames` walks the file once and keeps only the requested indices. cv2 is imported
lazily so the rest of the package (and the tests) stays importable without OpenCV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from safewatch.edge.extractor import DEFAULT_INPUT


def probe(path: str | Path) -> dict:
    """Frame count / fps / resolution, without decoding the whole file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def sample_frames(path: str | Path, indices, size: int = DEFAULT_INPUT) -> np.ndarray:
    """Return ``(len(indices), size, size, 3)`` float32 RGB in ``[0, 1]`` for the asked frames.

    Missing indices (video shorter than expected, or a decode hiccup) come back as zeros; the caller
    can detect them because the frame count is known from :func:`probe`. Raises if the file cannot
    be opened - silently returning zeros for a corrupt file is how a pipeline lies about itself.
    """
    import cv2

    wanted = sorted({int(i) for i in indices})
    if not wanted:
        raise ValueError("no frame indices requested")
    if min(wanted) < 0:
        raise ValueError("frame indices must be >= 0")

    out = np.zeros((len(wanted), size, size, 3), dtype=np.float32)
    slot = {index: position for position, index in enumerate(wanted)}
    remaining = set(wanted)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    cursor = 0
    last = wanted[-1]
    while remaining and cursor <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if cursor in remaining:
            remaining.discard(cursor)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out[slot[cursor]] = cv2.resize(rgb, (size, size),
                                           interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        cursor += 1
    cap.release()
    return out
