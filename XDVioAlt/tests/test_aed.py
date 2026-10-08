"""Tests for the acoustic-event detection layer (``safewatch.edge.aed``) and its wiring into the
alert card (``safewatch.explain.report``).

Pinned failure modes:

* the frame alignment is *measured*, not assumed: frame ``i`` covers ``[0.5i, 0.5i+0.96]`` s. A
  regression that resamples or shifts the window silently slides every named event by 0.5 s -
  caught by the tone-burst test when the model artefacts are present;
* a snippet past the audio's end must still produce tags (last frame repeated), never a dropped
  row - a dropped row desynchronises the tag stream from the snippet grid;
* the alert card must present YAMNet output as **corroboration**: the caveat flips with
  ``aed_tags``, and the "no incident event named" case must be said, not hidden.
"""
from __future__ import annotations

import numpy as np
import pytest

from safewatch.edge import aed

HAS_MODEL = aed.yamnet_available()


# --------------------------------------------------------------------------- #
# frame alignment (measured rule)
# --------------------------------------------------------------------------- #
def test_frame_overlapping_uses_the_measured_window():
    """Frame ``i`` = [0.5i, 0.5i+0.96] s; a snippet sees every frame whose window overlaps it.
    A 0.667 s snippet therefore spans 2-3 frames - the exact set the extractor pools when writing
    the tags onto the official 16-frame grid."""
    assert aed.frame_overlapping(0.0, 16 / 24.0, n_frames=10) == [0, 1]
    assert aed.frame_overlapping(16 / 24.0, 2 * 16 / 24.0, n_frames=10) == [0, 1, 2]
    assert aed.frame_overlapping(1.0, 1.6, n_frames=10) == [1, 2, 3]
    assert aed.frame_overlapping(0.0, 1.0, n_frames=0) == []


def test_frame_overlapping_falls_back_to_nearest_frame():
    """A snippet past every frame (very short audio) gets the nearest frame, never an empty set -
    otherwise the tag row would be missing and the snippet grid would desync."""
    idx = aed.frame_overlapping(9.0, 9.667, n_frames=2)
    assert idx == [1]


# --------------------------------------------------------------------------- #
# snippet pooling
# --------------------------------------------------------------------------- #
def test_snippet_tags_never_drops_a_row():
    """One tag row per snippet, whatever the clip length vs the audio length."""
    frames = np.tile(np.arange(16.0), (5, 521 // 16 + 1))[:, :521]      # 5 frames, 521 classes
    tags = aed.snippet_tags(frames, n_snippets=10, snippet_s=0.667)
    assert tags.shape == (10, 521)
    # snippets far past the audio end (10 snippets * 0.667 s = 6.7 s > 5 frames * 0.5 s = 2.5 s)
    # still get the last frame, not zeros
    assert np.allclose(tags[-1], frames[-1])


def test_snippet_tags_constant_frames_give_constant_tags():
    frames = np.full((8, 521), 0.37, dtype=np.float32)
    tags = aed.snippet_tags(frames, n_snippets=6, snippet_s=0.667)
    assert np.allclose(tags, 0.37)


def test_snippet_tags_empty_audio_gives_zeros():
    tags = aed.snippet_tags(np.zeros((0, 521), dtype=np.float32), n_snippets=4)
    assert tags.shape == (4, 521) and not tags.any()


# --------------------------------------------------------------------------- #
# incident tag set
# --------------------------------------------------------------------------- #
def test_incident_tag_set_covers_all_six_categories():
    idx, names = aed.incident_tag_indices()
    # every curated name must be in the 521-class map exactly once
    assert len(set(names)) == len(names)
    for category in ("explosion", "car_accident", "shooting", "fighting", "abuse", "riot"):
        assert any(label in names for label in aed.INCIDENT_TAG_SET[category])
    assert idx and all(0 <= i < 521 for i in idx)


def test_incident_energy_is_the_mean_of_the_curated_classes():
    names = list(aed._class_names())
    idx, _ = aed.incident_tag_indices(names)
    frames = np.zeros((3, 521), dtype=np.float32)
    for i in idx:
        frames[0, i] = 0.2
        frames[1, i] = 0.4
        frames[2, i] = 0.6
    energy = aed.incident_energy(frames)
    assert energy.shape == (3,)
    assert np.allclose(energy, [0.2, 0.4, 0.6])


def test_incident_tag_indices_rejects_a_missing_class():
    with pytest.raises(ValueError, match="class map lacks"):
        aed.incident_tag_indices(names=["Something", "Else"])


# --------------------------------------------------------------------------- #
# live model (needs the 16 MB artefact - skipped in slim checkouts)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_MODEL, reason="YAMNet ONNX artefact not present")
def test_yamnet_frames_shape_and_bounds():
    sr = aed.YAMNET_SAMPLE_RATE
    audio = np.sin(2 * np.pi * 440 * np.arange(0, 2.0, 1 / sr)).astype(np.float32) * 0.2
    frames = aed.yamnet_frames(audio)
    assert frames.shape == (4, 521)                    # 2 s / 0.5 s hop, measured
    assert frames.min() >= 0.0 and frames.max() <= 1.0


@pytest.mark.skipif(not HAS_MODEL, reason="YAMNet ONNX artefact not present")
def test_yamnet_alignment_tone_burst():
    """Pins the measured window alignment: a 440 Hz burst at 0.2-0.4 s must light up frame 0
    (window [0, 0.96]), not frame 1 or 2 - the same probe that established the rule."""
    sr = aed.YAMNET_SAMPLE_RATE
    t = np.arange(0, 2.0, 1 / sr)
    burst = np.sin(2 * np.pi * 440 * t) * 0.4 * ((t >= 0.2) & (t < 0.4))
    frames = aed.yamnet_frames(burst.astype(np.float32))
    assert frames.shape[0] == 4
    busy = aed._class_names().index("Busy signal")     # a 440 Hz tone reads as a phone tone
    assert frames[0, busy] > 0.5
    assert frames[0, busy] > frames[2, busy]


@pytest.mark.skipif(not HAS_MODEL, reason="YAMNet ONNX artefact not present")
def test_yamnet_is_deterministic():
    audio = np.random.default_rng(0).normal(size=16000).astype(np.float32) * 0.05
    assert np.allclose(aed.yamnet_frames(audio), aed.yamnet_frames(audio))


# --------------------------------------------------------------------------- #
# alert-card wiring (safewatch.explain.report)
# --------------------------------------------------------------------------- #
def _fake_context(aed_tags=None, snippet_probability=None) -> object:
    from safewatch.explain.report import ClipContext

    n = 20
    probability = snippet_probability if snippet_probability is not None \
        else np.concatenate([np.zeros(n - 5), np.full(5, 0.9)])
    return ClipContext(
        clip_id="fake", stride_frames=16.0, fps=24.0, n_snippets=n,
        snippet_probability=probability,
        category_probability={"explosion": 0.8},
        aed_tags=aed_tags,
    )


def test_audio_events_reports_named_peak_events_in_the_segment():
    from safewatch.explain.report import audio_events

    names = list(aed._class_names())
    explosion = names.index("Explosion")
    scream = names.index("Screaming")
    tags = np.zeros((20, 521), dtype=np.float32)
    tags[15, explosion] = 0.73       # inside the segment [15, 19]
    tags[17, scream] = 0.41
    context = _fake_context(aed_tags=tags)
    out = audio_events(context, segments=[(15, 19)], top_k=3)
    assert out["available"]
    events = out["segments"][0]["events"]
    assert events[0]["label"] == "Explosion" and abs(events[0]["probability"] - 0.73) < 1e-6
    assert events[0]["at_second"] == pytest.approx(15 * 16 / 24.0, abs=0.01)
    assert any(e["label"] == "Screaming" for e in events)
    # incident_energy over the segment: the two spiked snippets, mean over 16 curated classes
    assert out["segments"][0]["incident_energy"] > 0.0


def test_audio_events_names_nothing_when_nothing_fired():
    from safewatch.explain.report import audio_events

    tags = np.full((20, 521), 0.01, dtype=np.float32)
    out = audio_events(_fake_context(aed_tags=tags), segments=[(15, 19)], top_k=3)
    assert out["available"] and out["segments"][0]["events"] == []


def test_audio_events_degrades_gracefully_without_tags():
    from safewatch.explain.report import audio_events

    out = audio_events(_fake_context(), segments=[(15, 19)])
    assert not out["available"] and "note" in out


def _segment_config():
    """Config short enough that the fake 5-snippet (3.3 s) hot region [15, 19] becomes a segment."""
    from safewatch.explain.report import AlertConfig

    return AlertConfig(threshold=0.5, min_seconds=1.0, max_gap_seconds=2.5)


def test_build_alert_flips_the_caveat_with_aed_tags():
    from safewatch.explain.report import build_alert, describe_audio_phrases

    names = list(aed._class_names())
    explosion = names.index("Explosion")
    tags = np.zeros((20, 521), dtype=np.float32)
    tags[15, explosion] = 0.7
    config = _segment_config()
    with_tags = build_alert(_fake_context(aed_tags=tags), config)
    without_tags = build_alert(_fake_context(), config)
    caveat_with = [c for c in with_tags["caveats"] if "YAMNet" in c]
    caveat_without = [c for c in without_tags["caveats"] if "YAMNet" in c]
    assert caveat_with and not caveat_without
    assert "YAMNet" in with_tags["audio_evidence"]["acoustic_events"]["tagger"]
    phrases = describe_audio_phrases(with_tags["audio_evidence"])
    assert any("Explosion" in p and "YAMNet" in p for p in phrases)


def test_build_alert_says_out_loud_when_no_incident_sound_was_named():
    from safewatch.explain.report import build_alert

    tags = np.full((20, 521), 0.001, dtype=np.float32)   # no class crosses the 0.05 naming floor
    alert = build_alert(_fake_context(aed_tags=tags), _segment_config())
    phrases = alert["audio_evidence"]["descriptions"]
    assert any("no incident sound event named" in p for p in phrases)
