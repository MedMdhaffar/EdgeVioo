"""Unit tests for the SafeWatch evaluation harness.

Run:  .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from safewatch.eval import metrics as M


# --------------------------------------------------------------------------- #
# snippet <-> frame mapping
# --------------------------------------------------------------------------- #
def test_snippet_labels_respect_stride():
    labels = M.snippet_labels(13, 16, [(10, 40)])
    assert labels.tolist() == [1, 1, 1] + [0] * 10


def test_snippet_labels_clip_and_empty():
    assert M.snippet_labels(3, 16, [(0, 1000)]).tolist() == [1, 1, 1]
    assert M.snippet_labels(5, 16, []).tolist() == [0] * 5
    assert M.snippet_labels(5, 16, None).tolist() == [0] * 5
    assert M.snippet_labels(5, 16, [(0, 10), (0, 10)]).tolist() == [1] + [0] * 4


def test_frames_to_snippets_bounds():
    assert M.frames_to_snippets((0, 16), 16) == (0, 0)
    assert M.frames_to_snippets((0, 17), 16) == (0, 1)
    assert M.frames_to_snippets((31, 33), 16) == (1, 2)
    assert M.gt_segments_from_intervals([(0, 33)], 16) == [(0, 2)]


def test_snippet_labels_fractional_stride():
    """Real XD-Violence stride is ~63.3 frames (measured): snippet windows are rounded."""
    labels = M.snippet_labels(4, 63.3, [(0, 64)])
    assert labels.tolist() == [1, 1, 0, 0]
    assert M.frames_to_snippets((0, 64), 63.3) == (0, 1)
    assert M.frames_to_snippets((500, 900), 63.3) == (7, 14)


def test_prevalence():
    assert M.prevalence([1, 1, 0, 0]) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# AP metrics (expected values verified against sklearn in this environment)
# --------------------------------------------------------------------------- #
def test_perfect_ranking_gives_ap_one():
    labels = np.concatenate([np.zeros(6), np.ones(4)])
    scores = np.concatenate([np.linspace(0, 0.4, 6), np.linspace(0.6, 1.0, 4)])
    assert M.average_precision(labels, scores) == pytest.approx(1.0)
    assert M.pr_auc(labels, scores) == pytest.approx(1.0)


def test_worst_ranking_single_positive():
    labels = np.array([1.0] + [0.0] * 9)
    scores = np.linspace(0.0, 1.0, 10)  # the single positive gets the LOWEST score
    assert M.average_precision(labels, scores) == pytest.approx(0.1)


def test_constant_scores_tie_behaviour_is_documented():
    """average_precision is tie-safe (== prevalence); the official pr_auc is not.

    Measured here: prevalence 0.5 -> average_precision 0.5, pr_auc 0.75. The
    trapezoidal integration in the official protocol inflates scores when many
    predictions are tied, which is why both metrics are always reported.
    """
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert M.average_precision(labels, np.full(4, 0.5)) == pytest.approx(0.5)
    assert M.pr_auc(labels, np.full(4, 0.5)) == pytest.approx(0.75)


def test_degenerate_labels_are_nan():
    scores = np.array([0.1, 0.5, 0.6, 0.9])
    assert np.isnan(M.pr_auc(np.zeros(4), scores))
    assert np.isnan(M.average_precision(np.zeros(4), scores))


def test_protocols_and_aggregation():
    pred = {"a": np.array([0.1, 0.9]), "b": np.array([0.2, 0.3])}
    gt = {"a": np.array([0.0, 1.0]), "b": np.array([0.0, 0.0])}
    glob = M.evaluate(pred, gt, protocol="global", metric="average_precision")
    assert glob["n_snippets"] == 4 and glob["n_videos"] == 2
    assert glob["prevalence"] == pytest.approx(0.25)
    assert glob["ap"] == pytest.approx(1.0)
    per = M.evaluate(pred, gt, protocol="per_video", metric="average_precision")
    # video 'b' has no positive snippet -> NaN -> excluded from the mean
    assert per["ap"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        M.evaluate(pred, gt, protocol="nope")
    with pytest.raises(ValueError):
        M._resolve("bad-metric")


# --------------------------------------------------------------------------- #
# segmentation + localization
# --------------------------------------------------------------------------- #
def test_segments_from_scores_merges_and_filters():
    scores = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=float)
    assert M.segments_from_scores(scores, 0.5, min_length=1, max_gap=2) == [(2, 7), (10, 10)]
    assert M.segments_from_scores(scores, 0.5, min_length=2, max_gap=1) == [(2, 3), (5, 7)]
    assert M.segments_from_scores(np.zeros(4), 0.5) == []


def test_iou_inclusive():
    assert M.iou((0, 0), (0, 0)) == pytest.approx(1.0)
    assert M.iou((0, 1), (2, 3)) == pytest.approx(0.0)
    assert M.iou((0, 2), (2, 4)) == pytest.approx(1 / 5)


def test_localization_report_perfect_missed_and_late():
    gt = {"v": [(4, 8)]}
    perfect = M.localization_report({"v": [(4, 8)]}, gt, stride=16)
    assert perfect["n_pred_segments"] == 1 and perfect["n_gt_segments"] == 1
    assert perfect["f1@iou0.5"] == pytest.approx(1.0)
    assert perfect["mean_best_iou"] == pytest.approx(1.0)
    assert perfect["mean_detection_delay_snippets"] == pytest.approx(0.0)

    missed = M.localization_report({"v": [(40, 50)]}, gt, stride=16)
    assert missed["recall@iou0.5"] == 0
    assert missed["mean_best_iou"] == pytest.approx(0.0)

    # 1 snippet late -> IoU 4/7 = 0.57 -> still matched at tIoU 0.5
    late = M.localization_report({"v": [(5, 10)]}, gt, stride=16)
    assert late["mean_detection_delay_snippets"] == pytest.approx(1.0)
    assert late["f1@iou0.5"] == pytest.approx(1.0)

    # 2 snippets late -> IoU 3/7 = 0.43 -> already below the 0.5 threshold
    very_late = M.localization_report({"v": [(6, 10)]}, gt, stride=16)
    assert very_late["mean_detection_delay_snippets"] == pytest.approx(2.0)
    assert very_late["f1@iou0.5"] == pytest.approx(0.0)


def test_category_adaptive_segments():
    # A brief 2-snippet burst (0.8s on 16-frame 24fps grid, stride=0.67s)
    scores = np.zeros(20, dtype=float)
    scores[5:7] = 0.6  # 2 snippets

    # Standard 5.0s rule prunes this short burst
    std_segs = M.category_adaptive_segments(
        scores, category=None, stride_frames=16.0, fps=24.0, default_min_seconds=5.0)
    assert std_segs == []

    # Car accident adaptive rule (min 1.5s -> 2 snippets) catches it!
    car_segs = M.category_adaptive_segments(
        scores, category="car_accident", stride_frames=16.0, fps=24.0)
    assert car_segs == [(5, 6)]

    # Riot adaptive rule extends lookback window
    scores_riot = np.zeros(30, dtype=float)
    scores_riot[15:25] = 0.7  # 10 snippets
    riot_segs = M.category_adaptive_segments(
        scores_riot, category="riot", stride_frames=16.0, fps=24.0)
    assert len(riot_segs) == 1
    assert riot_segs[0][0] < 15  # lookback expanded start backwards
