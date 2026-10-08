"""Acoustic event detection (AED) via a frozen YAMNet (ONNX, CPU).

Fills the subject's "reconnaissance d'ev'ements acoustiques" requirement: instead of describing
audio with signal statistics only, the alert can *name* the sound events it hears (``Scream``,
``Car crash``, ``Explosion``, ``Skid``, ...) from a 521-class AudioSet-YouTube tagger.

Model: Google YAMNet (MobileNetV1, 4.2 M parameters), official weights, ONNX export
(``jafet21/yamnetonnx`` on Hugging Face, 16.1 MB, MIT/Apache lineage). It runs on
``onnxruntime`` already in the project venv - no TensorFlow, no GPU. Measured on the
project's CPU: ~50 ms for 6.7 s of audio (~8 ms per 0.96 s frame), i.e. comfortably
real-time for the edge/demo track.

Measured frame alignment (``scripts/check_vggish_space.py``-style, see
``docs/journal.md``): the ONNX emits ``max(1, n_samples // 8000)`` frames, and frame ``i``
covers roughly ``[0.5*i, 0.5*i + 0.96]`` seconds of the input (start-aligned 0.96 s window,
0.5 s hop, tail zero-padded). A snippet's tags are therefore the unweighted mean of the
frames whose window overlaps the snippet - documented, not assumed.

The model file is expected at ``data/features/aed/yamnet.onnx`` (the class map next to it).
It is data, not code: ``scripts/extract_aed_tags.py`` documents how to obtain it.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

AED_DIR = Path("data/features/aed")
YAMNET_PATH = AED_DIR / "yamnet.onnx"
CLASS_MAP_PATH = AED_DIR / "yamnet_class_map.csv"
TAGS_DIR = AED_DIR / "tags"
YAMNET_SAMPLE_RATE = 16000
YAMNET_FRAME_HOP_S = 0.5
YAMNET_FRAME_WINDOW_S = 0.96
YAMNET_FRAME_SAMPLES = int(YAMNET_FRAME_HOP_S * YAMNET_SAMPLE_RATE)

#: AudioSet classes that constitute an "incident-like" acoustic signature on the
#: XD-Violence task, grouped per subject category. Deliberately a *named, reviewed* set
#: (not a threshold trick): any class added here must make sense as evidence for the
#: category it is filed under. Kept as display names so the mapping is greppable.
INCIDENT_TAG_SET: dict[str, tuple[str, ...]] = {
    "explosion": ("Explosion", "Smash, crash", "Shatter"),
    "car_accident": ("Smash, crash", "Skidding", "Vehicle horn, car horn, honking", "Engine",
                     "Car alarm", "Engine knocking"),
    "shooting": ("Gunshot, gunfire", "Machine gun"),
    "fighting": ("Screaming", "Shout", "Smash, crash"),
    "abuse": ("Screaming", "Crying, sobbing", "Baby cry, infant cry"),
    "riot": ("Crowd", "Cheering", "Shout"),
}


@lru_cache(maxsize=1)
def _class_names() -> tuple[str, ...]:
    with CLASS_MAP_PATH.open(newline="", encoding="utf-8") as fh:
        return tuple(row["display_name"] for row in csv.DictReader(fh))


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    if not YAMNET_PATH.exists():
        raise FileNotFoundError(
            f"{YAMNET_PATH} not found - download it (see scripts/extract_aed_tags.py docstring)")
    return ort.InferenceSession(str(YAMNET_PATH), providers=["CPUExecutionProvider"])


def yamnet_available() -> bool:
    """True when the model artefacts are present (live mode degrades gracefully without them)."""
    return YAMNET_PATH.exists() and CLASS_MAP_PATH.exists()


def yamnet_frames(audio: np.ndarray, sample_rate: int = YAMNET_SAMPLE_RATE) -> np.ndarray:
    """Per-frame 521-class tag probabilities, shape ``(n_frames, 521)``.

    ``audio`` is a 1-D float32 waveform (any length >= 1 sample). ``n_frames`` is
    ``max(1, n_samples // 8000)`` and frame ``i`` covers ``[0.5i, 0.5i+0.96]`` s (see module
    docstring).
    """
    if audio.ndim != 1:
        raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros((0, 521), dtype=np.float32)
    scores = np.asarray(_session().run(None, {"waveform": audio})[0]).reshape(-1, 521)
    return scores.astype(np.float32)


def frame_overlapping(start_s: float, stop_s: float, n_frames: int) -> list[int]:
    """Indices of the frames whose [0.5i, 0.5i+0.96] window overlaps [start_s, stop_s).

    Frame ``j`` overlaps iff ``0.5j < stop_s`` and ``0.5j + 0.96 > start_s``, i.e.
    ``j < stop_s/0.5`` and ``j > (start_s - 0.96)/0.5`` - the second is a strict inequality,
    so the lower bound is ``floor((start_s - 0.96)/0.5) + 1`` (a frame whose window ends exactly
    at the snippet's start does NOT overlap it; an off-by-one here slid every pooled tag by half
    a frame - caught by ``tests/test_aed.py``).
    """
    if n_frames == 0:
        return []
    lo = max(0, int(np.floor((start_s - YAMNET_FRAME_WINDOW_S) / YAMNET_FRAME_HOP_S)) + 1)
    hi = min(n_frames - 1, int(np.floor(stop_s / YAMNET_FRAME_HOP_S - 1e-9)))
    idx = list(range(lo, hi + 1))
    if not idx:                      # tail of a very short clip: fall back to the nearest frame
        center = (start_s + stop_s) / 2.0
        idx = [min(n_frames - 1, max(0, int(round(center / YAMNET_FRAME_HOP_S))))]
    return idx


def snippet_tags(frames: np.ndarray, n_snippets: int, snippet_s: float = 16 / 24.0) -> np.ndarray:
    """Mean of the overlapping frames, per snippet - shape ``(n_snippets, 521)``.

    ``snippet_s`` is the snippet duration in seconds (``stride_frames / fps`` of the grid the
    clip was scored on). No overlap is dropped silently: a snippet past the audio's end gets the
    last frame repeated, so the caller can still see the tag stream.
    """
    n_frames = frames.shape[0]
    if n_frames == 0:
        return np.zeros((n_snippets, 521), dtype=np.float32)
    out = np.zeros((n_snippets, 521), dtype=np.float32)
    for i in range(n_snippets):
        idx = frame_overlapping(i * snippet_s, (i + 1) * snippet_s, n_frames)
        out[i] = frames[idx].mean(axis=0)
    return out


def tag_names() -> list[str]:
    """Display names, in the order of the 521 outputs."""
    return list(_class_names())


def incident_tag_indices(names: list[str] | None = None) -> tuple[list[int], list[str]]:
    """(indices, names) of the curated incident tag set - the only place it is defined."""
    names = names or list(_class_names())
    wanted: set[str] = set()
    for labels in INCIDENT_TAG_SET.values():
        wanted.update(labels)
    idx = [names.index(label) for label in sorted(wanted) if label in names]
    missing = wanted - set(names)
    if missing:
        raise ValueError(f"class map lacks incident classes: {sorted(missing)}")
    return idx, sorted(wanted)


def incident_energy(frames: np.ndarray, names: list[str] | None = None) -> np.ndarray:
    """Per-frame "incident audio" score: mean probability of the curated incident classes.

    This is the snippet-level analogue of the audio anomaly score the fusion model learns -
    but *named*: it only aggregates classes that plausibly accompany the six incident
    categories, so it can be quoted as evidence ("acoustic signature matches an incident").
    """
    idx, _ = incident_tag_indices(names)
    return frames[:, idx].mean(axis=1).astype(np.float32)
