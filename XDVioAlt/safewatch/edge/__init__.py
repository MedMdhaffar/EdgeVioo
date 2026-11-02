"""P7 (Edge AI): the lightweight extractor that replaces the frozen GPU backbone.

See :mod:`safewatch.edge.extractor` (model + budget), :mod:`safewatch.edge.video` (seek-free
frame sampling) and :mod:`safewatch.edge.audio` (audio extraction & quality estimation).
"""
from safewatch.edge import aed
from safewatch.edge.audio import (
    decode_audio,
    detect_acoustic_events,
    mel_patch_stats,
    quality_frames,
)
from safewatch.edge.extractor import (
    DEFAULT_INPUT,
    FEATURE_DIM,
    SNIPPET_FRAMES,
    MiniRGB,
    frame_grid,
    multi_frame_grid,
)
from safewatch.edge.video import probe, sample_frames, smooth_temporal_scores

__all__ = [
    "DEFAULT_INPUT",
    "FEATURE_DIM",
    "SNIPPET_FRAMES",
    "MiniRGB",
    "aed",
    "decode_audio",
    "detect_acoustic_events",
    "frame_grid",
    "multi_frame_grid",
    "mel_patch_stats",
    "probe",
    "quality_frames",
    "sample_frames",
    "smooth_temporal_scores",
]
