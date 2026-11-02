#!/usr/bin/env python
"""Evaluate the binary violence-vs-normal clip head on a held-out split.

This reports the dedicated ``clip_logits`` head, not the six auxiliary category logits or the
snippet-level localization curve. The threshold is fixed before evaluation (default 0.5); this
script never selects a threshold using test labels.

Example:
    .venv/bin/python scripts/eval_binary_classifier.py \
        --ckpt runs/<run>/ckpt_best.pt --test-csv data/lists_clip_k2_mstd/test.csv

For a source-disjoint final evaluation, exclude every source used for fitting or model selection::

    .venv/bin/python scripts/eval_binary_classifier.py --ckpt runs/<run>/ckpt_best.pt \
        --test-csv data/lists_binary_minirgb_sourceholdout/test.csv \
        --source-disjoint-from data/lists_binary_minirgb_sourceholdout/train.csv \
                               data/lists_binary_minirgb_sourceholdout/val.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# Allow direct invocation as ``python scripts/eval_binary_classifier.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safewatch.data.dataset import ClipDataset, load_rows
from safewatch.data.multimodal import (
    AUDIO_DIM,
    DEFAULT_QUALITY_DIR,
    DEFAULT_QUALITY_STATS,
    MultimodalClipDataset,
    audit_audio_rows,
    report_audio_preflight,
)
from safewatch.eval.runner import STRIDE_FRAMES, load_model


def source_of(clip_id: str) -> str:
    """Stable source identity used by the XD-Violence split audit (film/video before ``__#``)."""
    return clip_id.split("__#", 1)[0]


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Binary metrics at a threshold fixed before the reported split is examined."""
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fn = int(np.sum((y == 1) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "accuracy": float((tp + tn) / y.size) if y.size else None,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def threshold_maximising_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, dict]:
    """Choose the highest score threshold attaining validation-set maximum F1.

    The threshold candidates are exactly the observed model probabilities, so every possible
    prediction set under the ``p >= threshold`` decision rule is considered. The higher-threshold
    tie break retains precision when several operating points have equal F1.
    """
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError(
            "F1 threshold selection requires nonempty validation data with both labels")
    candidates = np.unique(p)
    best_threshold, best_metrics = float(candidates[0]), metrics_at_threshold(y, p, candidates[0])
    for threshold in candidates[1:]:
        metrics = metrics_at_threshold(y, p, float(threshold))
        if metrics["f1"] >= best_metrics["f1"]:
            best_threshold, best_metrics = float(threshold), metrics
    return best_threshold, best_metrics


def threshold_for_target_recall(y: np.ndarray, p: np.ndarray,
                                target_recall: float) -> tuple[float, dict]:
    """Choose the highest validation threshold that still reaches ``target_recall``.

    The highest-threshold tie break is deliberate: it preserves as much precision as possible
    while meeting the requested sensitivity.  This is a validation-only operating-point choice,
    never a way to tune a test result.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError("target-recall selection requires validation data with both labels")
    chosen_threshold, chosen_metrics = None, None
    for threshold in np.unique(p):
        metrics = metrics_at_threshold(y, p, float(threshold))
        if metrics["recall"] >= target_recall:
            chosen_threshold, chosen_metrics = float(threshold), metrics
    if chosen_threshold is None:  # defensive: threshold=min(p) predicts every positive.
        raise RuntimeError("no threshold reaches the requested recall")
    return chosen_threshold, chosen_metrics


def _dataset(test_csv: Path, cfg: dict, kind: str):
    modality = "both" if kind == "fusion" else cfg.get("modality", "visual")
    crop = cfg.get("crop", "mean")
    if modality == "visual":
        return ClipDataset(test_csv, max_snippets=None, train=False, crop=crop)
    stride = float(STRIDE_FRAMES.get(cfg.get("feature_set", "swin_rgb"), 64.0))
    dataset = MultimodalClipDataset(
        test_csv,
        cfg.get("audio_dir", "data/features/mel"),
        modality=modality,
        audio_stats=cfg.get("audio_stats"),
        quality_dir=cfg.get("quality_dir", DEFAULT_QUALITY_DIR),
        quality_stats=cfg.get("quality_stats", DEFAULT_QUALITY_STATS),
        audio_norm=cfg.get("audio_norm", "none"),
        audio_grid=cfg.get("audio_grid", "patch"),
        stride_frames=stride,
        max_snippets=None,
        train=False,
        crop=crop,
    )
    expected_dim = int(cfg.get("audio_dim", AUDIO_DIM)) if kind == "fusion" else int(
        cfg.get("in_dim") or AUDIO_DIM)
    report_audio_preflight(
        audit_audio_rows(dataset.rows, cfg.get("audio_dir", "data/features/mel"),
                         cfg.get("audio_grid", "patch"), expected_dim, stride,
                         quality_dir=cfg.get("quality_dir", DEFAULT_QUALITY_DIR)),
        "binary test evaluation",
    )
    return dataset


@torch.inference_mode()
def score(ckpt: Path, test_csv: Path, threshold: float, threads: int,
          source_disjoint_from: tuple[Path, ...] = ()) -> dict:
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, kind = load_model(ckpt, device)
    rows = load_rows(test_csv)
    dataset = _dataset(test_csv, cfg, kind)
    if len(dataset) != len(rows):
        raise RuntimeError(f"dataset/list mismatch: {len(dataset)} items vs {len(rows)} rows")

    excluded_sources = {
        source_of(str(row["clip_id"]))
        for path in source_disjoint_from
        for row in load_rows(path)
    }
    kept_indices = [index for index, row in enumerate(rows)
                    if source_of(str(row["clip_id"])) not in excluded_sources]
    if not kept_indices:
        raise RuntimeError("source-disjoint filter removed every test clip")

    ids: list[str] = []
    labels: list[float] = []
    probabilities: list[float] = []
    for index in kept_indices:
        item = dataset[index]
        if kind == "fusion":
            reliability = item.get("reliability")
            output = model(
                item["visual"][None].to(device),
                item["audio"][None].to(device),
                item["mask"][None].to(device),
                reliability[None].to(device) if reliability is not None else None,
            )
        else:
            features = item["audio"] if cfg.get("modality") == "audio" else item["features"]
            output = model(features[None].to(device), item["mask"][None].to(device))
        ids.append(str(item["clip_id"]))
        labels.append(float(item["binary"]))
        probabilities.append(float(torch.sigmoid(output["clip_logits"])[0].cpu()))

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    report = {
        "task": "binary_violence_detection",
        "checkpoint": str(ckpt),
        "test_csv": str(test_csv),
        "feature_set": cfg.get("feature_set"),
        "visual_dim": cfg.get("visual_dim", cfg.get("in_dim")),
        "audio_dim": cfg.get("audio_dim", 0),
        "stride_frames": STRIDE_FRAMES.get(cfg.get("feature_set", "swin_rgb"), 64.0),
        "n_clips": int(y.size),
        "n_positive": int(y.sum()),
        "n_negative": int(y.size - y.sum()),
        "source_disjoint": bool(source_disjoint_from),
        "source_disjoint_against": [str(path) for path in source_disjoint_from],
        "n_excluded_source_overlap": int(len(rows) - len(kept_indices)),
        "n_test_sources": len({source_of(str(rows[index]["clip_id"])) for index in kept_indices}),
        "positive_definition": "any of fighting, shooting, riot, abuse, car_accident, explosion",
        "score_source": "sigmoid(clip_logits)",
        "threshold": float(threshold),
        "threshold_policy": "fixed before test evaluation; not tuned on test labels",
        "metrics": metrics_at_threshold(y, p, threshold),
        "label_source": "binary field in the held-out split CSV",
    }
    return report, ids, y, p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-from-json", type=Path, default=None,
                        help="reuse selected_threshold from a prior validation metrics JSON")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--source-disjoint-from", nargs="*", type=Path, default=(),
                        help="CSV lists whose video sources must be excluded from --test-csv")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--select-f1-threshold", action="store_true",
                          help="select a threshold by maximum F1; use only on validation")
    selector.add_argument("--select-recall-threshold", type=float, default=None,
                          help="select highest threshold meeting recall target; validation only")
    parser.add_argument("--report-label", default=None,
                        help="optional safe suffix for metrics/predictions filenames")
    args = parser.parse_args()
    if args.threshold_from_json is not None:
        if not args.threshold_from_json.exists():
            parser.error(f"--threshold-from-json not found: {args.threshold_from_json}")
        try:
            selected = json.loads(args.threshold_from_json.read_text(encoding="utf-8"))[
                "threshold_selection"]["selected_threshold"]
            args.threshold = float(selected)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error("--threshold-from-json needs a validation threshold-selection metrics "
                         f"file: {exc}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    report, ids, labels, probabilities = score(
        args.ckpt, args.test_csv, args.threshold, args.threads,
        tuple(args.source_disjoint_from))
    out_dir = args.out_dir or args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "source_disjoint" if args.source_disjoint_from else "test"
    if args.select_f1_threshold:
        selected, selected_metrics = threshold_maximising_f1(labels, probabilities)
        report["threshold_selection"] = {
            "split": str(args.test_csv),
            "objective": "maximum_f1",
            "tie_break": "highest threshold",
            "selected_threshold": selected,
            "metrics_at_selected_threshold": selected_metrics,
            "warning": "This option is for validation only; do not use test labels to select "
                       "a threshold.",
        }
        suffix = "validation_threshold_selection"
    elif args.select_recall_threshold is not None:
        try:
            selected, selected_metrics = threshold_for_target_recall(
                labels, probabilities, args.select_recall_threshold)
        except ValueError as exc:
            parser.error(str(exc))
        report["threshold_selection"] = {
            "split": str(args.test_csv),
            "objective": "target_recall",
            "target_recall": args.select_recall_threshold,
            "tie_break": "highest threshold meeting target recall",
            "selected_threshold": selected,
            "metrics_at_selected_threshold": selected_metrics,
            "warning": "This option is for validation only; do not use test labels to select "
                       "a threshold.",
        }
        suffix = "validation_recall_threshold_selection"
    if args.report_label:
        if not args.report_label.replace("_", "").replace("-", "").isalnum():
            parser.error("--report-label may contain only letters, digits, hyphens and underscores")
        suffix = f"{suffix}_{args.report_label}"
    report_path = out_dir / f"binary_{suffix}_metrics.json"
    predictions_path = out_dir / f"binary_{suffix}_predictions.npz"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(predictions_path,
                        clip_ids=np.asarray(ids), labels=labels, probabilities=probabilities)
    print(json.dumps(report, indent=2))
    print(f"[ok] binary metrics -> {report_path}")
    print(f"[ok] per-clip scores -> {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
