"""Tests for P5: alert confidence calibration and the explanation card.

These pin the two things that would silently make the operator console lie:
* a temperature must actually reduce ECE on an over-confident model (and stay 1.0 when the model is
  already calibrated), otherwise the displayed percentage is decoration;
* every field the subject requires in an alert must be present and internally consistent - a card
  whose segment timestamps, probabilities or modality contributions disagree with the score curve it
  claims to explain is worse than no card.
"""
from __future__ import annotations

import numpy as np
import pytest

from safewatch.eval import calibration as C
from safewatch.explain import report as R


def _over_confident(n: int = 600, seed: int = 1):
    """A model that is confidently wrong on 25 % of the clips - the case calibration must fix."""
    rng = np.random.default_rng(seed)
    truth = (rng.random(n) < 0.3).astype(float)
    labels = np.where(rng.random(n) < 0.25, 1.0 - truth, truth)
    probabilities = np.where(truth == 1, 0.93, 0.07)
    return probabilities, labels


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_temperature_fixes_an_over_confident_model():
    probabilities, labels = _over_confident()
    before = C.expected_calibration_error(probabilities, labels)
    fitted = C.fit(probabilities, labels, split="unit")
    after = C.expected_calibration_error(fitted.probability(probabilities), labels)

    assert fitted.temperature > 1.0                     # softening is the right direction here
    assert after < before / 2                        # a real improvement, not a rounding artefact
    assert fitted.metrics["nll_after"] < fitted.metrics["nll_before"]
    assert fitted.fitted_on["split"] == "unit"


def test_temperature_is_near_one_when_already_calibrated():
    rng = np.random.default_rng(0)
    labels = (rng.random(4000) < 0.4).astype(float)
    probabilities = np.clip(0.4 + rng.normal(0, 0.05, labels.size), 0.01, 0.99)
    fitted = C.fit(probabilities, np.where(rng.random(labels.size) < probabilities, 1.0, 0.0))
    assert 0.5 < fitted.temperature < 2.0                # no wild rescaling of a sane model


def test_calibration_map_roundtrip_and_absence(tmp_path):
    fitted = C.fit(*_over_confident(), split="unit")
    path = fitted.save(tmp_path / "calibration.json")
    reloaded = C.CalibrationMap.load(path)

    assert reloaded is not None
    assert reloaded.temperature == pytest.approx(fitted.temperature, rel=1e-9)
    assert reloaded.fitted_on == fitted.fitted_on
    missing = C.CalibrationMap.load(tmp_path / "missing.json")
    assert missing is None                           # callers must say "uncalibrated"


def test_calibration_probability_is_monotone_and_bounded():
    fitted = C.CalibrationMap(temperature=2.0)
    raw = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    calibrated = fitted.probability(raw)
    assert np.all(np.diff(calibrated) > 0)               # ranking must survive calibration
    assert calibrated.min() > 0 and calibrated.max() < 1
    logits = np.log(raw / (1.0 - raw))                   # same inputs expressed as logits
    assert np.allclose(fitted.probability(logits, already_probability=False),
                       fitted.probability(raw))

def _per_class_case(n: int = 400, seed: int = 7):
    """Three categories with different failure modes, to pin per-class behaviour.

    * ``good``   - over-confident but fittable -> a temperature must be found;
    * ``absent`` - zero positives: the ``abuse`` situation on the official val split;
    * ``calm``   - already sane, so its temperature should stay near 1.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((n, 3), dtype=np.float64)
    truth = rng.random(n) < 0.25
    # mirror ``_over_confident``: a quarter of the labels are flipped, so a 0.9 score is wrong a
    # quarter of the time -> genuinely over-confident -> temperature must soften (T > 1)
    labels[:, 0] = np.where(rng.random(n) < 0.25, 1.0 - truth, truth)
    labels[:, 2] = rng.random(n) < 0.5
    probabilities = np.zeros((n, 3), dtype=np.float64)
    probabilities[:, 0] = np.where(truth == 1, 0.9, 0.1)
    probabilities[:, 1] = 0.05                                    # a head that never fires
    probabilities[:, 2] = np.clip(0.5 + rng.normal(0, 0.05, n), 0.01, 0.99)
    return probabilities, labels


def test_per_class_calibration_skips_a_class_absent_from_the_fitting_split():
    """``abuse`` has no positive in the official val split, so that head cannot be fitted.

    The point is that this is *reported* (``unfitted`` list, identity temperature) rather than
    silently handing back a T that was never learned.
    """
    probabilities, labels = _per_class_case()
    fitted = C.fit_per_class(probabilities, labels, ("good", "absent", "calm"), split="unit")

    assert fitted.unfitted == ["absent"]
    assert fitted.temperature("absent") == 1.0
    assert "absent" not in fitted.maps
    assert fitted.metrics["per_class"]["absent"]["fitted"] is False
    assert np.allclose(fitted.probability(probabilities[:, 1], "absent"), probabilities[:, 1])


def test_per_class_calibration_fixes_the_heads_it_can_fit():
    probabilities, labels = _per_class_case()
    fitted = C.fit_per_class(probabilities, labels, ("good", "absent", "calm"), split="unit")
    good = fitted.metrics["per_class"]["good"]

    assert good["temperature"] > 1.0                     # over-confident -> soften
    assert good["ece_after"] < good["ece_before"]
    assert good["n_pos"] == int(labels[:, 0].sum())
    # the macro is taken over fitted heads only, so the unfitted one cannot dilute it
    expected = (good["ece_before"] + fitted.metrics["per_class"]["calm"]["ece_before"]) / 2
    assert fitted.metrics["macro_ece_before"] == pytest.approx(expected)
    assert fitted.fitted_on["fitted_categories"] == ["good", "calm"]


def test_per_class_calibration_roundtrip_and_shape_validation(tmp_path):
    probabilities, labels = _per_class_case()
    fitted = C.fit_per_class(probabilities, labels, ("good", "absent", "calm"), split="unit")
    reloaded = C.PerClassCalibration.load(fitted.save(tmp_path / "per_class.json"))

    assert reloaded is not None
    assert reloaded.unfitted == fitted.unfitted
    assert reloaded.temperature("good") == pytest.approx(fitted.temperature("good"), rel=1e-9)
    assert C.PerClassCalibration.load(tmp_path / "missing.json") is None
    with pytest.raises(ValueError):                      # probabilities/labels must line up
        C.fit_per_class(np.zeros((5, 3)), np.zeros((5, 2)), ("a", "b", "c"))
    with pytest.raises(ValueError):                      # ... and match the category list
        C.fit_per_class(np.zeros((5, 3)), np.zeros((5, 3)), ("a", "b"))




def test_reliability_bins_cover_every_sample_once():
    probabilities, labels = _over_confident(n=200)
    bins = C.reliability_bins(probabilities, labels, n_bins=10)
    assert sum(bins["count"]) == 200
    assert len(bins["bin_edges"]) == 11
    assert len(bins["confidence"]) == len(bins["accuracy"]) == 10
    with pytest.raises(ValueError):                       # a 1-sample bin cannot hold a bar
        C.reliability_bins(np.array([0.5, 0.5]), np.array([1.0]))


# --------------------------------------------------------------------------- #
# alert card (the subject's seven required fields)
# --------------------------------------------------------------------------- #
def _context(**overrides) -> R.ClipContext:
    """A 20-snippet clip (2.667 s/snippet => ~53 s) with one clear anomaly around snippets 6-8."""
    probability = np.full(20, 0.05)
    probability[6:9] = 0.9
    probability[7] = 0.97
    reliability = np.zeros((20, 7), dtype=np.float32)
    reliability[:, 1] = -30.0                    # rms_db baseline
    reliability[6:9, 1] = -12.0                  # +18 dB inside the event
    reliability[6:9, 2] = 0.6                    # broadband
    reliability[6:9, 4] = 0.03                   # slight saturation
    defaults = dict(
        clip_id="clip_demo", stride_frames=64.0, fps=24.0, n_snippets=20,
        snippet_probability=probability,
        attention=np.array([0.01] * 6 + [0.2, 0.35, 0.15] + [0.04] * 11),
        category_probability={"fighting": 0.8, "shooting": 0.2, "riot": 0.1, "abuse": 0.01,
                              "car_accident": 0.05, "explosion": 0.3},
        reliability=reliability,
        snippet_visual=np.clip(probability + 0.02, 0, 1),
        snippet_audio=np.clip(probability - 0.3, 0, 1),
        meta={"drop_one": {"audio": {"mean_probability_intact": 0.3,
                                    "mean_probability_without": 0.3, "delta": 0.0}}},
        duration_s=53.3, video_path=None)
    defaults.update(overrides)
    return R.ClipContext(**defaults)


def test_alert_has_every_field_the_subject_requires():
    card = R.build_alert(_context(), R.AlertConfig(language="fr"),
                         C.CalibrationMap(temperature=2.0))

    for field in ("clip_id", "category", "segments", "visual_evidence", "audio_evidence",
                  "modality_contribution", "confidence", "text", "caveats"):
        assert field in card, f"missing {field}"

    assert card["requires_human_verification"] is True
    assert card["category"]["label"] == "fighting"
    assert len(card["segments"]) == 1
    assert card["confidence"]["calibrated"] is True


def test_segments_are_in_seconds_and_inside_the_clip():
    card = R.build_alert(_context(), R.AlertConfig(threshold=0.5, min_seconds=2.0))

    assert card["segments"], "the synthetic anomaly must produce a segment"
    segment = card["segments"][0]
    assert segment["start_s"] == pytest.approx(6 * 64 / 24, abs=1e-6)
    assert segment["end_s"] == pytest.approx(9 * 64 / 24, abs=1e-6)
    assert segment["peak_probability"] == pytest.approx(0.97)
    assert segment["n_snippets_above_threshold"] == 3
    assert 0 <= segment["start_s"] < segment["end_s"] <= card["duration_s"]


def test_min_seconds_filters_short_segments_across_grids():
    """The same 2 s rule must mean the same duration on the 64-frame and the 16-frame grid."""
    mirror = R.build_alert(_context(), R.AlertConfig(min_seconds=8.0))     # 8 s = 3 snippets here
    assert len(mirror["segments"]) == 1                       # the anomaly is exactly 3 snippets

    official = R.build_alert(_context(stride_frames=16.0),
                             R.AlertConfig(min_seconds=8.0))               # 8 s = 12 snippets here
    assert official["segments"] == [], "8 s is longer than the anomaly at a 16-frame stride"


def test_visual_and_audio_evidence_are_derived_not_invented():
    card = R.build_alert(_context(), R.AlertConfig(top_k_evidence=3))

    top = card["visual_evidence"]["top_snippets"]
    assert len(top) == 3
    assert top[0]["snippet"] == 7 and top[0]["probability"] == pytest.approx(0.97)
    assert top[0]["attention"] == pytest.approx(0.35)
    assert card["visual_evidence"]["peak_attention"]["snippet"] == 7
    assert 0 <= card["visual_evidence"]["attention_entropy_normalised"] <= 1

    audio = card["audio_evidence"]
    assert audio["available"] is True
    assert audio["segment_vs_background_db"] == pytest.approx(18.0, abs=0.01)
    assert audio["clipping_ratio_max"] == pytest.approx(0.03)
    assert any("large bande" in phrase or "broadband" in phrase for phrase in audio["descriptions"])


def test_audio_evidence_says_nothing_when_channels_are_missing():
    card = R.build_alert(_context(reliability=None))
    assert card["audio_evidence"]["available"] is False
    assert "note" in card["audio_evidence"]                        # no invented sound events


def test_modality_contribution_reports_curves_and_drop_one():
    contribution = R.build_alert(_context())["modality_contribution"]

    assert contribution["available"] is True
    assert contribution["mean_probability"]["visual"] > contribution["mean_probability"]["audio"]
    assert set(contribution["drop_one"]) == {"audio"}


def test_confidence_is_labelled_uncalibrated_when_no_temperature_exists():
    card = R.build_alert(_context(), calibration=None)
    confidence = card["confidence"]

    assert confidence["calibrated"] is False
    assert "calibration.json" in confidence["note"]
    assert "non calibree" in card["text"]


def test_text_contains_the_alerts_facts_in_both_languages():
    context = _context()
    for language in ("fr", "en"):
        card = R.build_alert(context, R.AlertConfig(language=language),
                             C.CalibrationMap(temperature=2.0))
        text = card["text"]
        assert "00:16" in text                                     # segment start (snippet 6)
        assert "fighting" in text and "80%" in text
        assert card["config"]["language"] == language
    assert "Human verification" in R.build_alert(
        context, R.AlertConfig(language="en"), C.CalibrationMap(temperature=2.0))["text"]


def test_no_segment_means_normal_clip_and_still_explains():
    flat = _context(snippet_probability=np.full(20, 0.02))
    card = R.build_alert(flat, R.AlertConfig(threshold=0.5))
    assert card["segments"] == []
    assert "threshold" in card["text"].lower() or "seuil" in card["text"].lower()


def test_summarise_is_console_ready():
    card = R.build_alert(_context(), R.AlertConfig(language="fr"))
    summary = R.summarise(card)
    assert "clip_demo" in summary and "segment" in summary
    assert len(summary.splitlines()) < 12                     # must fit on a console screen


def test_ranking_is_by_importance_and_stable_under_saturated_scores():
    """Tied scores must not reshuffle between runs - scores saturate at 1.0 in practice."""
    plateau = _context(snippet_probability=np.array([0.05] * 5 + [1.0] * 4 + [0.05] * 11))

    with_attention = R.build_alert(
        plateau, R.AlertConfig(top_k_evidence=3))["visual_evidence"]["top_snippets"]
    # all four plateau snippets score 1.0, so attention (0.35 > 0.2 > 0.15) decides the rank
    assert [s["snippet"] for s in with_attention] == [7, 6, 8]

    without_attention = R.build_alert(
        _context(snippet_probability=plateau.snippet_probability, attention=None),
        R.AlertConfig(top_k_evidence=3))["visual_evidence"]["top_snippets"]
    assert [s["snippet"] for s in without_attention] == [5, 6, 7]  # ties: earliest snippet first

    assert with_attention == R.build_alert(
        plateau, R.AlertConfig(top_k_evidence=3))["visual_evidence"]["top_snippets"]


def test_clip_score_reconstruction_uses_the_top_k_rule():
    curve = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    assert C.clip_scores_from_snippet_curve(curve, k=2) == pytest.approx((0.9 + 0.8) / 2)
    assert C.clip_scores_from_snippet_curve(curve, k=99) == pytest.approx(curve.mean())
    assert np.isnan(C.clip_scores_from_snippet_curve(np.array([])))


def test_isotonic_fixes_an_over_confident_model():
    """The non-parametric rung: when temperature is one scalar, isotonic has the whole curve."""
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 1.6, size=400)
    labels = (_sig(logits / 2.4) > rng.uniform(size=400)).astype(float)
    raw = np.clip(_sig(logits), 1e-4, 1 - 1e-4)

    before = C.expected_calibration_error(raw, labels)
    payload = C.fit_isotonic(raw, labels, split="val")
    after = C.expected_calibration_error(C.apply_isotonic(raw, payload["mapping"]), labels)
    assert payload["method"] == "isotonic"
    assert payload["metrics"]["ece_before"] == pytest.approx(before)
    assert payload["metrics"]["ece_after"] == pytest.approx(after)
    assert after < before                                        # monotone map wins here
    assert payload["metrics"]["nll_after"] <= payload["metrics"]["nll_before"] + 1e-9


def test_isotonic_refuses_a_single_class_split():
    with pytest.raises(ValueError):
        C.fit_isotonic(np.array([0.2, 0.4, 0.6]), np.zeros(3))


def test_apply_isotonic_clips_the_ends_and_stays_bounded():
    mapping = {"x": [0.0, 0.3, 0.7, 1.0], "y": [0.02, 0.1, 0.6, 0.95]}
    out = C.apply_isotonic(np.array([-1.0, 0.3, 0.5, 2.0]), mapping)
    assert out[0] == pytest.approx(0.02)                          # below range -> left clip
    assert out[1] == pytest.approx(0.1)                           # exact breakpoint
    assert out[2] == pytest.approx(0.1 + 0.5 * 0.5)               # linear between breaks (0.35)
    assert out[3] == pytest.approx(0.95)                          # above range -> right clip
    assert np.all(np.diff(out) >= -1e-12)                         # monotone
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_per_class_isotonic_keeps_the_absent_class_unfitted():
    """Same contract as the temperature variant: a class with one label stays identity, not hidden."""
    rng = np.random.default_rng(1)
    n, n_cat = 300, 6
    probs = rng.uniform(0.05, 0.95, size=(n, n_cat))
    labels = (probs > rng.uniform(size=(n, n_cat))).astype(float)
    labels[:, 3] = 0.0                                            # category 3: single-class target

    payload = C.fit_per_class_isotonic(probs, labels, [f"c{i}" for i in range(n_cat)])
    assert payload["method"] == "isotonic"
    assert payload["unfitted"] == ["c3"]
    assert "c3" not in payload["mappings"]
    assert payload["metrics"]["per_class"]["c3"]["fitted"] is False
    for i in range(n_cat):
        if i == 3:
            continue
        entry = payload["metrics"]["per_class"][f"c{i}"]
        assert entry["fitted"] is True
        assert entry["ece_after"] <= entry["ece_before"] + 1e-9
