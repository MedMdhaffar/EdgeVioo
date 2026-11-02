"""Frame sampling for the edge track: decode **sequentially**, never seek.

Measured on this machine (``docs/journal.md``, Session 15):

* sequential decode: **1800-2900 fps** (a 4-minute clip in ~3.2 s);
* ``CAP_PROP_POS_FRAMES`` seek: **25-50 ms per frame** (the decoder restarts from the previous
  keyframe), i.e. ~10x the cost of reading the frames it skipped.

So :func:`iter_sample_frames` walks the file once and yields small batches of requested indices.
The compatibility wrapper :func:`sample_frames` still collects all batches for offline callers.
cv2 is imported lazily so the rest of the package stays importable without OpenCV.
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


def iter_sample_frames(path: str | Path, indices, size: int = DEFAULT_INPUT,
                       batch_size: int = 32, crop_corridor: bool = False,
                       foveal_zoom: bool = False, clahe: bool = False):
    """Yield ``(indices, frames)`` batches in temporal order using bounded frame memory.

    Each frame batch is float32 RGB in ``[0, 1]`` with shape ``(B, size, size, 3)``. Requested
    indices are sorted and deduplicated, matching :func:`sample_frames`. Missing frames after a
    decode failure are yielded as zeros so batch positions stay aligned with the requested grid.
    Only one decode pass is used; frames are never sought individually.
    If ``crop_corridor=True``, crops the forward driving corridor (middle 60% width, 15%-85% height)
    to suppress peripheral windshield glare and passing vehicle lights.
    If ``foveal_zoom=True``, blends the wide corridor with a central horizon zoom (vanishing point)
    to maintain high spatial fidelity for distant vehicles and hazards (>50m).
    If ``clahe=True``, applies adaptive contrast equalization in the LAB L-channel to eliminate
    headlight sweeps, glare, and extreme darks under night-time dashcam conditions.
    """
    import cv2

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    wanted = sorted({int(i) for i in indices})
    if not wanted:
        raise ValueError("no frame indices requested")
    if min(wanted) < 0:
        raise ValueError("frame indices must be >= 0")

    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    cursor = 0
    decode_ok = True
    try:
        for start in range(0, len(wanted), batch_size):
            batch_indices = wanted[start:start + batch_size]
            out = np.zeros((len(batch_indices), size, size, 3), dtype=np.float32)
            slots = {index: position for position, index in enumerate(batch_indices)}
            if decode_ok:
                while cursor <= batch_indices[-1]:
                    ok, frame = cap.read()
                    if not ok:
                        decode_ok = False
                        break
                    position = slots.get(cursor)
                    if position is not None:
                        h, w = frame.shape[:2]
                        if foveal_zoom:
                            # Foveal horizon zoom: 50% central corridor focus (vanishing point)
                            # blended 60% wide + 40% zoomed horizon to capture distant vehicle hazards
                            fy1, fy2 = int(0.25 * h), int(0.75 * h)
                            fx1, fx2 = int(0.25 * w), int(0.75 * w)
                            fovea = frame[fy1:fy2, fx1:fx2]
                            fovea_resized = cv2.resize(fovea, (size, size), interpolation=cv2.INTER_AREA)

                            if crop_corridor:
                                y1, y2 = int(0.15 * h), int(0.85 * h)
                                x1, x2 = int(0.20 * w), int(0.80 * w)
                                frame = frame[y1:y2, x1:x2]
                            wide_resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                            frame = cv2.addWeighted(wide_resized, 0.60, fovea_resized, 0.40, 0)
                        else:
                            if crop_corridor:
                                y1, y2 = int(0.15 * h), int(0.85 * h)
                                x1, x2 = int(0.20 * w), int(0.80 * w)
                                frame = frame[y1:y2, x1:x2]
                            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if clahe_op is not None:
                            # Adaptive histogram equalization on lightness to suppress headlight glare
                            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                            lab[:, :, 0] = clahe_op.apply(lab[:, :, 0])
                            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

                        out[position] = rgb.astype(np.float32) / 255.0
                    cursor += 1
            yield np.asarray(batch_indices, dtype=np.int64), out
    finally:
        cap.release()


def sample_frames(path: str | Path, indices, size: int = DEFAULT_INPUT,
                  crop_corridor: bool = False, foveal_zoom: bool = False,
                  clahe: bool = False) -> np.ndarray:
    """Collect and return all requested frames as ``(T,size,size,3)`` float32 RGB in ``[0,1]``.

    Kept for offline scripts needing a single array. Live inference should use
    :func:`iter_sample_frames` and process each batch before requesting the next one.
    """
    batches = [frames for _batch_indices, frames in
               iter_sample_frames(path, indices, size=size, crop_corridor=crop_corridor,
                                  foveal_zoom=foveal_zoom, clahe=clahe)]
    return np.concatenate(batches, axis=0)


def compute_snippet_motion_dynamics(sampled_frames: np.ndarray, n_snippets: int, frames_per_snippet: int = 4) -> dict[str, np.ndarray]:
    """Compute temporal motion differential and shock dynamics across sampled frames.

    Args:
        sampled_frames: Array of shape ``(n_snippets * frames_per_snippet, H, W, C)`` in ``[0, 1]``
                        OR list of batch arrays.
        n_snippets: Number of temporal snippets.
        frames_per_snippet: Number of sub-frames sampled per snippet (typically 4).

    Returns:
        dict with:
            - ``mean_motion``: Mean frame differential per snippet, shape ``(n_snippets,)``.
            - ``burst_shock``: Intra-snippet differential spread (max - min step), shape ``(n_snippets,)``.
            - ``motion_energy``: Total motion differential variance across snippet sub-steps.
    """
    if isinstance(sampled_frames, (list, tuple)):
        if not sampled_frames:
            return {
                "mean_motion": np.zeros(0, dtype=np.float32),
                "burst_shock": np.zeros(0, dtype=np.float32),
                "motion_energy": np.zeros(0, dtype=np.float32),
            }
        sampled_frames = np.concatenate(sampled_frames, axis=0)

    if sampled_frames.size == 0 or n_snippets <= 0:
        return {
            "mean_motion": np.zeros(0, dtype=np.float32),
            "burst_shock": np.zeros(0, dtype=np.float32),
            "motion_energy": np.zeros(0, dtype=np.float32),
        }

    total_frames = n_snippets * frames_per_snippet
    frames = sampled_frames[:total_frames].reshape(n_snippets, frames_per_snippet, *sampled_frames.shape[1:])
    if frames_per_snippet > 1:
        # Consecutive frame differences within each snippet: (n_snippets, frames_per_snippet - 1, H, W, C)
        diffs = np.abs(frames[:, 1:] - frames[:, :-1]).mean(axis=(2, 3, 4))
        mean_motion = diffs.mean(axis=1).astype(np.float32)
        burst_shock = (diffs.max(axis=1) - diffs.min(axis=1)).astype(np.float32)
        motion_energy = diffs.std(axis=1).astype(np.float32)
    else:
        mean_motion = np.zeros(n_snippets, dtype=np.float32)
        burst_shock = np.zeros(n_snippets, dtype=np.float32)
        motion_energy = np.zeros(n_snippets, dtype=np.float32)

    return {
        "mean_motion": mean_motion,
        "burst_shock": burst_shock,
        "motion_energy": motion_energy,
    }


def smooth_temporal_scores(scores: np.ndarray, alpha_rise: float = 0.85,
                           alpha_decay: float = 0.25) -> np.ndarray:
    """Asymmetric exponential moving average (EMA) filter on sequential anomaly scores.

    Fast attack (``alpha_rise``) guarantees rapid alert responsiveness within <0.5s of an impact,
    while smooth decay (``alpha_decay``) eliminates momentary single-frame drops and fluttering
    without lagging initial detections.
    """
    if scores.ndim == 0 or scores.size <= 1:
        return scores.astype(np.float32, copy=False)
    smoothed = np.empty_like(scores, dtype=np.float32)
    current = float(scores[0])
    smoothed[0] = current
    for i in range(1, len(scores)):
        val = float(scores[i])
        alpha = alpha_rise if val >= current else alpha_decay
        current = alpha * val + (1.0 - alpha) * current
        smoothed[i] = current
    return smoothed

