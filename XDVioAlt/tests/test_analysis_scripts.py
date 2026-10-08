"""Regression tests for the analysis scripts added in Session 10 (Tier 1 audit).

Each of these scripts answers a question the report depends on, and each has a silent failure mode:

* ``per_class_localization`` - the segment rule must be converted from **seconds** to snippets on
  the *run's* grid, otherwise the same snippet count means 5.3 s on the mirror grid and 1.3 s on the
  official one (the trap that produced a bogus "localisation collapsed" reading);
* ``per_class_localization.report_for`` - restricting to a category must restrict both the
  predictions *and* the ground truth, or a category would be scored against corpus-wide events;
* ``class_diagnostic`` - the TP/FP counts at threshold and the co-occurrence table are what justify
  the "abuse is starved, not broken" conclusion, so they are pinned here.
* ``seed_margin`` - ``--metrics-suffix`` must select the metrics file (offline vs causal); a silent
  fall-back would report offline numbers as streaming and hide the seed-correction it produced.
* ``ladder_seed_check`` - the clean AP of a run comes from ``test_metrics.json`` *or* from ``level
  0`` of a robustness sweep (the gate-replication seeds only wrote the latter); reading the ladder
  off one seed is what hid that ``adaptive``'s 3-seed mean is *below* the visual baseline.

Run:  .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import class_diagnostic as CD  # noqa: E402
import ladder_seed_check as LSC  # noqa: E402
import per_class_localization as PCL  # noqa: E402
import seed_margin as SM  # noqa: E402

from safewatch.data.dataset import CATEGORIES  # noqa: E402


def test_segment_rule_seconds_give_different_snippet_counts_per_grid():
    """The same wall-clock rule is 2 snippets on the mirror grid and 8 on the official one."""
    from safewatch.eval import metrics as M

    assert M.snippets_for_seconds(5.333, 64.0, 24.0) == 2
    assert M.snippets_for_seconds(5.333, 16.0, 24.0) == 8
    assert M.snippets_for_seconds(2.667, 16.0, 24.0, minimum=0) == 4


def test_report_for_subsets_predictions_and_ground_truth_together():
    """A clip outside ``ids`` must contribute neither a prediction nor a GT segment."""
    scores = {"hot": np.ones(20, dtype=np.float32), "cold": np.zeros(20, dtype=np.float32)}
    intervals = {"hot": [(0, 320)], "cold": [(0, 320)]}   # 320 frames / 16 = snippets 0..19

    only_hot = PCL.report_for(["hot"], scores, intervals, stride=16.0, threshold=0.5,
                              min_length=8, max_gap=4, fps=24.0)
    assert only_hot["n_pred_segments"] == 1
    assert only_hot["n_gt_segments"] == 1
    assert only_hot["mean_best_iou"] == pytest.approx(1.0)

    only_cold = PCL.report_for(["cold"], scores, intervals, stride=16.0, threshold=0.5,
                               min_length=8, max_gap=4, fps=24.0)
    assert only_cold["n_pred_segments"] == 0
    assert only_cold["n_gt_segments"] == 1
    assert np.isnan(only_cold["mean_best_iou"])


def test_min_length_in_seconds_controls_which_segments_survive():
    """A 10-snippet burst is 6.7 s on the official grid: kept at a 5.33 s rule, dropped at 8 s."""
    from safewatch.eval import metrics as M

    scores = np.concatenate([np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32)])

    assert M.snippets_for_seconds(5.333, 16.0, 24.0) == 8      # 10 snippets survive 8
    assert M.snippets_for_seconds(8.0, 16.0, 24.0) == 12       # 10 snippets do not survive 12
    assert len(M.segments_from_scores(scores, 0.5, min_length=8, max_gap=4)) == 1
    assert M.segments_from_scores(scores, 0.5, min_length=12, max_gap=4) == []


def test_resolve_stride_reads_the_sibling_metrics_file(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "test_metrics.json").write_text(json.dumps({"stride_frames": 16.0}),
                                               encoding="utf-8")
    predictions = run_dir / "predictions.npz"
    assert PCL.resolve_stride(predictions, None, None) == 16.0
    assert PCL.resolve_stride(predictions, 64.0, None) == 64.0   # explicit flag wins


def test_resolve_stride_fails_loudly_without_a_grid(tmp_path: Path):
    """Guessing the grid silently destroys AP/segments, so there is no default."""
    predictions = tmp_path / "predictions.npz"
    with pytest.raises(SystemExit):
        PCL.resolve_stride(predictions, None, None)


def test_class_diagnostic_counts_threshold_hits_and_co_occurrence():
    rows = [
        {"clip_id": "a", "labels": "abuse"},                 # positive, head barely fires
        {"clip_id": "b", "labels": "abuse|fighting"},        # positive, mixed clip
        {"clip_id": "c", "labels": "riot"},                  # negative
    ]
    abuse = CATEGORIES.index("abuse")
    multilogits = {
        "a": np.array([0.1, 0.1, 0.1, 0.05, 0.1, 0.1]),
        "b": np.array([0.2, 0.1, 0.1, 0.90, 0.1, 0.1]),
        "c": np.array([0.1, 0.1, 0.9, 0.02, 0.1, 0.1]),
    }
    rep = CD.diagnostic(rows, multilogits, "abuse", top=2)

    assert rep["n_positives"] == 2 and rep["n_negatives"] == 1
    assert rep["n_positive_above_0.5"] == 1          # only clip "b" clears the threshold
    assert rep["n_negative_above_0.5"] == 0
    assert rep["co_occurrence_in_positives"] == {"abuse": 2, "fighting": 1}
    assert rep["lowest_scoring_positives"][0]["clip_id"] == "a"
    assert multilogits["a"][abuse] == pytest.approx(0.05)


def _write_metrics(run_dir: Path, ap_global: float, ap_clip: float, suffix: str = "") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"test_metrics{suffix}.json").write_text(json.dumps({
        "ap": {"global__pr_auc": ap_global, "per_video__average_precision": ap_clip},
        "macro_per_class_ap": 0.5,
        "localization": {"mean_best_iou": 0.3},
    }), encoding="utf-8")


def test_seed_margin_suffix_switches_the_metrics_file(tmp_path: Path):
    """``--metrics-suffix`` must change *which* file is read, not silently fall back.

    The P4 seed check compares causal against causal; if the suffix were ignored the script would
    print the **offline** numbers as if they were streaming, and the correction it produced (the
    localisation effect flipping sign across seeds) would have been invisible. So the contract is
    pinned: a suffix with no matching file yields *nothing*, never the offline metrics.
    """
    runs = tmp_path / "runs"
    for seed, causal_ap in ((43, 0.68), (44, 0.69)):
        cross = runs / f"2026-01-01_p3vggish30ep_cross_s{seed}"
        visual = runs / f"2026-01-01_p2vggish30ep_visual_s{seed}"
        _write_metrics(cross, 0.77, 0.79)
        _write_metrics(cross, causal_ap, 0.60, suffix="_causal8")
        _write_metrics(visual, 0.73, 0.71)
        _write_metrics(visual, 0.65, 0.61, suffix="_causal8")

    offline = SM.collect(runs, "")
    causal = SM.collect(runs, "_causal8")
    assert offline["p3vggish30ep_cross"][43]["ap_global"] == pytest.approx(0.77)
    assert causal["p3vggish30ep_cross"][43]["ap_global"] == pytest.approx(0.68)
    assert causal["p3vggish30ep_cross"][44]["ap_global"] == pytest.approx(0.69)
    assert set(causal["p3vggish30ep_cross"]) == {43, 44}       # seeds paired, not merged
    assert SM.collect(runs, "_causal9") == {}                 # no file -> empty, NOT offline data


def test_ladder_seed_check_reads_clean_ap_from_either_source(tmp_path: Path):
    """A run's clean AP lives in ``test_metrics.json`` - or, for the gate-replication seeds, only in
    ``level 0`` of a robustness sweep. Reading the ladder off the seed-42 column is what hid that
    ``adaptive``'s 3-seed mean sits below the visual baseline, so both sources must be found, and
    "level 0" must mean the entry with ``level == 0.0`` - not simply the first list element.
    """
    runs = tmp_path / "runs"
    scored = runs / "2026-01-01_p3vggish30ep_cross"
    _write_metrics(scored, 0.7745, 0.7961)
    sweep_only = runs / "2026-01-01_p3vggish30ep_adaptive_s43"
    sweep_only.mkdir(parents=True)
    (sweep_only / "robustness_occlusion_visual_blind.json").write_text(json.dumps({
        "levels": [{"level": 30.0, "ap_global": 0.05},          # deliberately out of order: the
                   {"level": 0.0, "ap_global": 0.6319},         # clean entry is NOT levels[0]
                   {"level": 15.0, "ap_global": 0.10}]}), encoding="utf-8")

    assert LSC.clean_ap(scored) == (pytest.approx(0.7745), "test_metrics")
    assert LSC.clean_ap(sweep_only) == (pytest.approx(0.6319), "sweep_L0")
    assert LSC.clean_ap(runs / "2026-01-01_missing") is None
