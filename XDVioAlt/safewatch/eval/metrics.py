"""Evaluation metrics for weakly-supervised audio-visual anomaly detection.

Two AP protocols coexist in the XD-Violence literature and they are **not** the same:

* ``pr_auc`` - one precision-recall curve over the snippets of all test videos
  concatenated, scored with ``sklearn.metrics.auc``. This is what the official
  XDVioDet release does (HL-Net RGB+audio+flow AP = 0.7864).
* ``average_precision`` - area under the step-interpolated PR curve
  (``sklearn.metrics.average_precision_score``), the usual convention.

Both can be aggregated ``global`` (all snippets pooled) or ``per_video``
(mean of the 800 per-video APs, as reported by VadCLIP / MSBT / MAVD).

Everything works at **snippet** level: a snippet is one feature vector spanning
``stride`` video frames (see ``docs/dataset.md``), so frame-level ground truth is
converted before scoring.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


# --------------------------------------------------------------------------- #
# snippet <-> frame mapping
# --------------------------------------------------------------------------- #
def snippet_labels(n_snippets: int, stride: float, intervals) -> np.ndarray:
    """Binary per-snippet labels.

    Snippet ``i`` covers frames ``[round(i*stride), round((i+1)*stride))`` and is
    labelled anomalous iff that window overlaps any annotated ``(start, end)``
    interval. ``stride`` is a float because the actual stride is clip-dependent
    (measured: ~63.3 frames = ~2.64 s at 24 fps, i.e. T = ceil(nb_frames / 64)).
    """
    labels = np.zeros(int(n_snippets), dtype=np.float32)
    for start, end in intervals or []:
        start, end = float(start), float(end)
        if end <= start:
            continue
        first = max(0, int(np.floor(start / stride)))
        last = min(int(n_snippets) - 1, int(np.ceil(end / stride)) - 1)
        if last >= first:
            labels[first:last + 1] = 1.0
    return labels


def frames_to_snippets(interval, stride: float) -> tuple[int, int]:
    """Frame interval -> inclusive snippet range (float-safe)."""
    start, end = float(interval[0]), float(interval[1])
    first = int(np.floor(start / stride))
    return first, max(first, int(np.ceil(end / stride)) - 1)


def gt_segments_from_intervals(intervals, stride: int) -> list[tuple[int, int]]:
    return [frames_to_snippets(iv, stride) for iv in intervals or []]


def snippets_for_seconds(seconds: float, stride_frames: float, fps: float = 24.0,
                         minimum: int = 1) -> int:
    """Convert a duration in **seconds** to a whole number of snippets on the measured grid.

    Segment rules were expressed in snippets, which silently means different durations per grid
    (``min_length=2`` is 5.3 s on the mirror's 64-frame grid but only 1.3 s on the official 16-frame
    one - the reason the official run predicted 1 865 segments for 1 238 events). Anything published
    must be expressed in seconds and converted here once.
    """
    return max(minimum, int(round(seconds * fps / stride_frames)))


def prevalence(labels) -> float:
    """Fraction of positive snippets - the AP of an uninformative scorer."""
    labels = np.asarray(labels).ravel()
    return float(labels.mean()) if labels.size else float("nan")


# --------------------------------------------------------------------------- #
# AP metrics
# --------------------------------------------------------------------------- #
def pr_auc(labels, scores) -> float:
    """AP as computed by the official XDVioDet code (``auc`` over the PR curve).

    NOTE: with tied scores (e.g. a constant predictor) this yields
    ``(prevalence + 1) / 2`` instead of ``prevalence`` - a known artefact of
    trapezoidal integration on a degenerate curve. Use ``average_precision``
    for tie-safe behaviour.
    """
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def average_precision(labels, scores) -> float:
    """Tie-safe average precision (step-interpolated PR curve)."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    return float(average_precision_score(labels, scores))


def _resolve(metric: str):
    if metric == "pr_auc":
        return pr_auc
    if metric == "average_precision":
        return average_precision
    raise ValueError(f"unknown metric {metric!r}")


def _stack(pred_by_video: dict, gt_by_video: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(pred_by_video)
    scores = np.concatenate([np.asarray(pred_by_video[k]).ravel() for k in keys])
    labels = np.concatenate([np.asarray(gt_by_video[k]).ravel() for k in keys])
    return labels, scores


def ap_global(pred_by_video: dict, gt_by_video: dict, metric: str = "pr_auc") -> float:
    """Pooled protocol: one PR curve over the snippets of all videos."""
    labels, scores = _stack(pred_by_video, gt_by_video)
    return _resolve(metric)(labels, scores)


def ap_per_video(pred_by_video: dict, gt_by_video: dict,
                 metric: str = "pr_auc") -> tuple[float, dict[str, float]]:
    """Per-video protocol: mean over per-video APs (degenerate videos excluded)."""
    fn = _resolve(metric)
    per_video = {k: fn(gt_by_video[k], pred_by_video[k]) for k in sorted(pred_by_video)}
    valid = [v for v in per_video.values() if not np.isnan(v)]
    mean_ap = float(np.mean(valid)) if valid else float("nan")
    return mean_ap, per_video


def evaluate(pred_by_video: dict, gt_by_video: dict, protocol: str = "global",
             metric: str = "pr_auc") -> dict:
    """Aggregate AP under one protocol. Returns a JSON-serialisable dict."""
    if protocol == "global":
        labels, scores = _stack(pred_by_video, gt_by_video)
        return {
            "protocol": "global",
            "metric": metric,
            "ap": _resolve(metric)(labels, scores),
            "n_videos": len(pred_by_video),
            "n_snippets": int(labels.size),
            "prevalence": prevalence(labels),
        }
    if protocol == "per_video":
        mean_ap, per_video = ap_per_video(pred_by_video, gt_by_video, metric)
        return {
            "protocol": "per_video",
            "metric": metric,
            "ap": mean_ap,
            "n_videos": len(pred_by_video),
            "per_video": per_video,
        }
    raise ValueError(f"unknown protocol {protocol!r}; expected 'global' or 'per_video'")


# --------------------------------------------------------------------------- #
# temporal localization & category-adaptive thresholds
# --------------------------------------------------------------------------- #
DEFAULT_CATEGORY_THRESHOLDS: dict[str, float] = {
    "abuse": 0.10,
    "car_accident": 0.35,
    "explosion": 0.35,
    "fighting": 0.45,
    "riot": 0.40,
    "shooting": 0.40,
}

DEFAULT_CATEGORY_TEMPORAL_PARAMS: dict[str, dict[str, float]] = {
    "car_accident": {"min_seconds": 1.5, "max_gap_seconds": 1.5, "lookback_seconds": 0.0},
    "explosion": {"min_seconds": 1.5, "max_gap_seconds": 1.5, "lookback_seconds": 0.0},
    "riot": {"min_seconds": 4.0, "max_gap_seconds": 3.0, "lookback_seconds": 4.0},
    "fighting": {"min_seconds": 3.0, "max_gap_seconds": 2.5, "lookback_seconds": 2.0},
    "shooting": {"min_seconds": 2.0, "max_gap_seconds": 2.0, "lookback_seconds": 0.0},
    "abuse": {"min_seconds": 3.0, "max_gap_seconds": 2.5, "lookback_seconds": 2.0},
}


def segments_from_scores(scores, threshold: float = 0.5, min_length: int = 2,
                         max_gap: int = 1) -> list[tuple[int, int]]:
    """Threshold a per-snippet score curve into inclusive ``(start, end)`` segments.

    ``max_gap`` merges segments separated by fewer than ``max_gap`` empty snippets;
    ``min_length`` drops segments shorter than that many snippets.
    """
    binary = np.asarray(scores).ravel() >= threshold
    raw: list[list[int]] = []
    start = None
    for i, flag in enumerate(binary):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            raw.append([start, i - 1])
            start = None
    if start is not None:
        raw.append([start, len(binary) - 1])

    merged: list[list[int]] = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] - 1 < max_gap:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return [(s, e) for s, e in merged if e - s + 1 >= min_length]


def category_adaptive_segments(
    scores,
    category: str | None = None,
    stride_frames: float = 16.0,
    fps: float = 24.0,
    default_threshold: float = 0.5,
    default_min_seconds: float = 5.0,
    default_max_gap_seconds: float = 2.5,
) -> list[tuple[int, int]]:
    """Segment an anomaly curve with category-calibrated thresholds and temporal rules."""
    params = DEFAULT_CATEGORY_TEMPORAL_PARAMS.get(category, {}) if category else {}
    min_s = float(params.get("min_seconds", default_min_seconds))
    max_gap_s = float(params.get("max_gap_seconds", default_max_gap_seconds))
    lookback_s = float(params.get("lookback_seconds", 0.0))

    threshold = (DEFAULT_CATEGORY_THRESHOLDS.get(category, default_threshold)
                 if category else default_threshold)

    min_snippets = snippets_for_seconds(min_s, stride_frames, fps, minimum=1)
    max_gap_snippets = snippets_for_seconds(max_gap_s, stride_frames, fps, minimum=0)
    lookback_snippets = (snippets_for_seconds(lookback_s, stride_frames, fps, minimum=0)
                         if lookback_s > 0 else 0)

    segments = segments_from_scores(
        scores, threshold=threshold, min_length=min_snippets, max_gap=max_gap_snippets)
    if lookback_snippets > 0:
        expanded = [(max(0, s - lookback_snippets), e) for s, e in segments]
        merged_exp: list[tuple[int, int]] = []
        for seg in expanded:
            if merged_exp and seg[0] <= merged_exp[-1][1]:
                merged_exp[-1] = (merged_exp[-1][0], max(merged_exp[-1][1], seg[1]))
            else:
                merged_exp.append(seg)
        segments = merged_exp
    return segments


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Intersection over union of two inclusive snippet intervals."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union else 0.0


def _greedy_match(preds, gts, threshold: float) -> int:
    """Greedy highest-IoU-first matching; returns the number of true positives."""
    pairs = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(preds)
                    for gi, g in enumerate(gts)), reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    n_match = 0
    for score, pi, gi in pairs:
        if score < threshold or pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        n_match += 1
    return n_match


def localization_report(pred_segments: dict, gt_segments: dict,
                        thresholds=(0.1, 0.2, 0.3, 0.4, 0.5), stride: int = 1,
                        fps: float = 24.0) -> dict:
    """Segment-level P/R/F1 at several tIoU, plus mean best-IoU and detection delay.

    Segments are inclusive snippet ranges. Delays are reported in snippets, in frames
    (``snippets * stride``) and in seconds (``frames / fps``) because mixing those units is a
    classic silent error: the first version of this function multiplied snippets by the frame
    stride and *called* the result seconds (300 "s" instead of 12.5 s).
    """
    n_pred = sum(len(v) for v in pred_segments.values())
    n_gt = sum(len(v) for v in gt_segments.values())
    tp = {t: 0 for t in thresholds}
    best_ious: list[float] = []
    delays: list[float] = []

    for vid, gts in gt_segments.items():
        preds = pred_segments.get(vid, [])
        for p in preds:
            if not gts:
                continue
            ious = [iou(p, g) for g in gts]
            j = int(np.argmax(ious))
            best_ious.append(ious[j])
            if ious[j] > 0:
                delays.append(p[0] - gts[j][0])
        for t in thresholds:
            tp[t] += _greedy_match(preds, gts, t)

    out: dict = {
        "n_pred_segments": n_pred,
        "n_gt_segments": n_gt,
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
        "mean_detection_delay_snippets": float(np.mean(delays)) if delays else float("nan"),
        "mean_detection_delay_frames": (float(np.mean(delays)) * stride
                                        if delays else float("nan")),
        "mean_detection_delay_seconds": (float(np.mean(delays)) * stride / fps
                                         if delays else float("nan")),
    }
    for t in thresholds:
        precision = tp[t] / n_pred if n_pred else float("nan")
        recall = tp[t] / n_gt if n_gt else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else 0.0)
        out[f"precision@iou{t:g}"] = precision
        out[f"recall@iou{t:g}"] = recall
        out[f"f1@iou{t:g}"] = f1
    return out
