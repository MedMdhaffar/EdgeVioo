#!/usr/bin/env python
"""Evaluate binary fusion performance as the visual feature stream is degraded.

This is a representation-level robustness study: it modifies precomputed CLIP descriptors, not
video pixels. It compares the same fusion checkpoint with audio available versus audio explicitly
marked unavailable, on the run's source-held-out validation split. The threshold is fixed at 0.5.

Example:
    .venv/bin/python scripts/eval_binary_visual_robustness.py \
        --ckpt runs/<run>/ckpt_best.pt --val-csv data/lists_sourceholdout/val.csv
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
from safewatch.eval import robustness as RB  # noqa: E402
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
def evaluate(ckpt: Path, val_csv: Path, threshold: float, threads: int, seed: int):
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, kind = load_model(ckpt, device)
    if kind != "fusion":
        raise ValueError(f"expected a fusion checkpoint, got {kind}")
    dataset = _dataset(val_csv, cfg, kind)
    if len(dataset) != len(load_rows(val_csv)):
        raise RuntimeError("dataset/list mismatch")

    # Attenuation scales visual descriptors toward zero; occlusion zeroes one contiguous span.
    # The 64-frame grid is 2.667 seconds per snippet at 24 fps for this run.
    conditions = [("clean", None, None)]
    conditions.extend((f"visual_attenuation_{level:.2f}", "attenuation", level)
                      for level in (0.75, 0.50, 0.25, 0.0))
    conditions.extend((f"visual_occlusion_{seconds:.2f}s", "occlusion", seconds)
                      for seconds in (2.667, 5.333, 10.667))

    ids: list[str] = []
    labels: list[int] = []
    probs = {name: {"fusion": [], "video_only": []} for name, _, _ in conditions}
    for item in dataset:
        visual = item["visual"][None].to(device)
        audio = item["audio"][None].to(device)
        mask = item["mask"][None].to(device)
        rel_item = item.get("reliability")
        reliability = rel_item[None].to(device) if rel_item is not None else None

        ids.append(str(item["clip_id"]))
        labels.append(int(item["binary"]))
        for name, kind_name, level in conditions:
            v, a, rel = visual, audio, reliability
            if kind_name is not None:
                hook = RB.build_perturbation(kind_name, float(level), "visual", cfg,
                                             seed=seed, flag_reliability=False)
                v, a, rel = hook(v, a, rel)
            output = model(v, a, mask, rel)
            probs[name]["fusion"].append(
                float(torch.sigmoid(output["clip_logits"])[0].cpu()))

            # Compare against the same model receiving the degraded video alone.
            v_only, a_zero, rel_only = R.drop_modality(v, a, rel, "audio")
            output = model(v_only, a_zero, mask, rel_only)
            probs[name]["video_only"].append(
                float(torch.sigmoid(output["clip_logits"])[0].cpu()))

    y = np.asarray(labels, dtype=np.int64)
    probabilities = {
        name: {mode: np.asarray(values, dtype=np.float64) for mode, values in pair.items()}
        for name, pair in probs.items()
    }
    results = {}
    for name, _, _ in conditions:
        fused = _metrics(y, probabilities[name]["fusion"], threshold)
        video_only = _metrics(y, probabilities[name]["video_only"], threshold)
        results[name] = {
            "fusion": fused,
            "video_only": video_only,
            "fusion_delta_vs_video_only": {
                metric: fused[metric] - video_only[metric]
                for metric in ("average_precision", "roc_auc", "accuracy", "f1")
                if fused[metric] is not None and video_only[metric] is not None
            },
        }

    report = {
        "task": "binary_visual_representation_robustness",
        "checkpoint": str(ckpt),
        "validation_csv": str(val_csv),
        "n_clips": int(y.size),
        "n_positive": int(y.sum()),
        "n_negative": int(y.size - y.sum()),
        "score_source": "sigmoid(clip_logits)",
        "threshold": float(threshold),
        "threshold_policy": "fixed at 0.5; not tuned on this validation sweep",
        "comparison": "same checkpoint with audio present (fusion) vs audio marked unavailable",
        "visual_degradation_space": "precomputed visual feature representation, not video pixels",
        "degradations": {
            "attenuation": "multiply visual descriptors by level toward zero; 0 is no visual signal",
            "occlusion": "zero one deterministic contiguous feature window per clip; durations in seconds",
        },
        "seed": seed,
        "conditions": results,
    }
    return report, ids, y, probabilities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    report, ids, labels, probabilities = evaluate(
        args.ckpt, args.val_csv, args.threshold, args.threads, args.seed)
    out_dir = args.out_dir or args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "binary_visual_robustness_val.json"
    predictions_path = out_dir / "binary_visual_robustness_val_predictions.npz"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    flat = {f"{condition}__{mode}": values
            for condition, modes in probabilities.items()
            for mode, values in modes.items()}
    np.savez_compressed(predictions_path, clip_ids=np.asarray(ids), labels=labels, **flat)
    print(json.dumps(report, indent=2))
    print(f"[ok] visual robustness metrics -> {metrics_path}")
    print(f"[ok] per-clip scores -> {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
