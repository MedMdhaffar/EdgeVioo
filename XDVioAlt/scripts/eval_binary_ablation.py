#!/usr/bin/env python
"""Fixed-threshold binary clip-head ablation for an audio-visual fusion checkpoint.

Scores the same held-out clips with both modalities, with audio marked unavailable, and with
video marked unavailable. The missing stream is zeroed and its reliability channels are set to the
same degraded sentinel used during modality dropout in training. No thresholds are fitted here.

Example:
    .venv/bin/python scripts/eval_binary_ablation.py --ckpt runs/<run>/ckpt_best.pt \
        --test-csv data/lists_clip_k2_mstd/test.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safewatch.data import reliability as R  # noqa: E402
from safewatch.data.dataset import load_rows  # noqa: E402
from safewatch.eval.runner import load_model  # noqa: E402
from scripts.eval_binary_classifier import _dataset  # noqa: E402


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fn = int(np.sum((y == 1) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "accuracy": float(np.mean(pred == y)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


@torch.inference_mode()
def evaluate(ckpt: Path, test_csv: Path, threshold: float, threads: int):
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, kind = load_model(ckpt, device)
    if kind != "fusion":
        raise ValueError(f"expected a fusion checkpoint, got {kind}")
    dataset = _dataset(test_csv, cfg, kind)
    if len(dataset) != len(load_rows(test_csv)):
        raise RuntimeError("dataset/list mismatch")

    ids: list[str] = []
    labels: list[int] = []
    probs = {"fused": [], "video_only": [], "audio_only": []}
    for item in dataset:
        visual = item["visual"][None].to(device)
        audio = item["audio"][None].to(device)
        mask = item["mask"][None].to(device)
        rel_item = item.get("reliability")
        reliability = rel_item[None].to(device) if rel_item is not None else None

        output = model(visual, audio, mask, reliability)
        probs["fused"].append(float(torch.sigmoid(output["clip_logits"])[0].cpu()))

        # Match the model's trained missing-modality path: zero features and flag the absent stream.
        v_drop, a_drop, r_drop = R.drop_modality(visual, audio, reliability, "audio")
        output = model(v_drop, a_drop, mask, r_drop)
        probs["video_only"].append(float(torch.sigmoid(output["clip_logits"])[0].cpu()))

        v_drop, a_drop, r_drop = R.drop_modality(visual, audio, reliability, "visual")
        output = model(v_drop, a_drop, mask, r_drop)
        probs["audio_only"].append(float(torch.sigmoid(output["clip_logits"])[0].cpu()))

        ids.append(str(item["clip_id"]))
        labels.append(int(item["binary"]))

    y = np.asarray(labels, dtype=np.int64)
    probabilities = {key: np.asarray(value, dtype=np.float64)
                     for key, value in probs.items()}
    base = _metrics(y, probabilities["fused"], threshold)
    report = {
        "task": "binary_violence_detection_modality_ablation",
        "checkpoint": str(ckpt),
        "test_csv": str(test_csv),
        "n_clips": int(y.size),
        "n_positive": int(y.sum()),
        "n_negative": int(y.size - y.sum()),
        "score_source": "sigmoid(clip_logits)",
        "threshold": float(threshold),
        "threshold_policy": "fixed before test evaluation; not tuned on test labels",
        "missing_modality_policy": "zero missing features and set reliability channels to degraded sentinel",
        "conditions": {},
        "deltas_vs_fused": {},
    }
    for condition, p in probabilities.items():
        condition_metrics = _metrics(y, p, threshold)
        report["conditions"][condition] = condition_metrics
        if condition != "fused":
            report["deltas_vs_fused"][condition] = {
                metric: (condition_metrics[metric] - base[metric])
                for metric in ("average_precision", "roc_auc", "accuracy", "f1")
                if condition_metrics[metric] is not None and base[metric] is not None
            }
    return report, ids, y, probabilities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    report, ids, labels, probabilities = evaluate(
        args.ckpt, args.test_csv, args.threshold, args.threads)
    out_dir = args.out_dir or args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "binary_modality_ablation.json"
    predictions_path = out_dir / "binary_modality_ablation_predictions.npz"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(predictions_path, clip_ids=np.asarray(ids), labels=labels,
                        **{f"probabilities_{key}": value
                           for key, value in probabilities.items()})
    print(json.dumps(report, indent=2))
    print(f"[ok] modality ablation metrics -> {metrics_path}")
    print(f"[ok] per-clip scores -> {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
