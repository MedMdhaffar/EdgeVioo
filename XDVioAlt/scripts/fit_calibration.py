#!/usr/bin/env python
"""Fit the alert confidence calibration (temperature scaling) on a checkpoint's **val** split.

Writes ``runs/<run>/calibration.json``:

    {"temperature": 2.31,
     "fitted_on": {"split": "val", "source": "clip_logits", "n_samples": 434},
     "metrics": {"ece_before": ..., "ece_after": ..., "nll_before": ..., "nll_after": ...},
     "reliability_val": {...},          # reliability-diagram bins for the report figure
     "test_reconstructed": {...}}       # with --test, see the caveat below

Why val: the temperature is a *fitted* parameter, so fitting it on test labels would be leakage.
Val is the split already used for model selection (`docs/protocol.md` §2).

The optional ``--test`` block rebuilds a clip-level score from the snippet curve the runner saves,
with the same top-k rule the training objective uses
(`calibration.clip_scores_from_snippet_curve`), because the runner stores snippet curves and not
clip logits. It is labelled ``reconstructed`` and its ECE is quoted with the caveat that the
temperature came from val.

Usage:
    .venv/bin/python scripts/fit_calibration.py \\
        --ckpt runs/2026-09-29_p3full_late/ckpt_best.pt --test
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.eval import calibration as C  # noqa: E402
from safewatch.eval.runner import load_model  # noqa: E402


def _config(cfg: dict, threads: int):
    """Rebuild the training config object from a checkpoint's config dict.

    Going through the real config classes (and their dataset builders below) is deliberate: the val
    samples must be preprocessed exactly as they were during training, otherwise the fitted
    temperature describes a different input pipeline than the one the alerts will use.
    """
    from safewatch import train as mil_module
    from safewatch import train_fusion as fusion_module

    module = fusion_module if cfg.get("fusion_mode") else mil_module
    cls = fusion_module.FusionConfig if cfg.get("fusion_mode") else mil_module.TrainConfig
    allowed = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in cfg.items() if k in allowed}
    kwargs["max_snippets"] = None        # full-length inference, as the scorer does
    kwargs["workers"] = 0
    kwargs["threads"] = threads
    return cls(**kwargs), module, bool(cfg.get("fusion_mode"))


def _val_split_outputs(ckpt: str, lists_dir: str | None, threads: int):
    """Val split, both heads: ``(clip_probs, clip_labels, multi_probs, multi_labels)``.

    The two heads are different calibration problems: ``clip_logits`` scores "is this clip
    anomalous", ``multi_logits`` names the *category* the alert card shows. One pass fills both.
    """
    import torch

    torch.set_num_threads(threads)
    model, cfg, kind = load_model(ckpt, torch.device("cpu"))
    config, module, is_fusion = _config(cfg, threads)
    csv_path = Path(getattr(config, "lists_dir", None) or lists_dir or "data/lists") / "val.csv"
    if not csv_path.exists():
        raise SystemExit(f"[calib] no validation split at {csv_path}")
    dataset = (module._dataset(config, csv_path, train=False) if is_fusion
               else module._make_dataset(config, csv_path, train=False))

    scores: list[float] = []
    labels: list[float] = []
    multi_probs: list[np.ndarray] = []
    multi_labels: list[np.ndarray] = []
    for item in dataset:
        with torch.no_grad():
            if is_fusion:
                reliability = item.get("reliability")
                out = model(item["visual"][None], item["audio"][None], item["mask"][None],
                            reliability[None] if reliability is not None else None)
            else:
                out = model(item["visual"][None], item["mask"][None])
        scores.append(float(torch.sigmoid(out["clip_logits"])[0]))
        labels.append(float(item["binary"]))
        multi_probs.append(torch.sigmoid(out["multi_logits"])[0].numpy())
        multi_labels.append(item["multi"].numpy())
    return (np.asarray(scores), np.asarray(labels),
            np.stack(multi_probs), np.stack(multi_labels))


def val_clip_probabilities(ckpt: str, lists_dir: str | None,
                           threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip-level probabilities + weak labels on the validation split of the checkpoint's run."""
    clip_probs, clip_labels, _multi_probs, _multi_labels = _val_split_outputs(
        ckpt, lists_dir, threads)
    return clip_probs, clip_labels



def test_clip_multilabel(ckpt: str, lists_dir: str | None,
                         threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-category probabilities + weak labels on the **test** split.

    Goes through ``runner.score_clips`` - the single scoring path, so these are the probabilities
    the deployed scorer would emit - and reads the categories from the test list's ``labels``
    column, exactly as ``compute_metrics`` does for the per-class AP.
    """
    import torch

    from safewatch.data.dataset import CATEGORIES, load_rows
    from safewatch.eval.runner import default_test_csv, score_clips

    torch.set_num_threads(threads)
    _model, cfg, _kind = load_model(ckpt, torch.device("cpu"))
    test_csv = Path(lists_dir) / "test.csv" if lists_dir else default_test_csv(cfg)
    _scores, _valid, multi = score_clips(ckpt, test_csv, torch.device("cpu"))
    rows = load_rows(test_csv)
    labels = np.asarray([[1.0 if code in (row["labels"] or "").split("|") else 0.0
                          for code in CATEGORIES] for row in rows])
    probs = np.stack([multi[row["clip_id"]] for row in rows])
    return probs, labels


def _run_per_class(args) -> int:
    """Fit one calibrator per category; report ECE on val (fitted) and test (evaluated).

    P5's open question from Session 11: the *global* temperature improved NLL but not ECE. The alert
    card shows a category, so the multi-label head is the head that needs calibrating - and a class
    absent from the fitting split (``abuse``: zero val positives) cannot be calibrated at all. Both
    facts are printed, not smoothed over.

    ``--method isotonic`` fits the non-parametric monotone map instead of one temperature, the
    untried rung of the standard ladder when temperature scaling measurably fails (Session 14).
    """
    from safewatch.data.dataset import CATEGORIES

    _clip, _clip_y, multi_probs, multi_labels = _val_split_outputs(
        args.ckpt, args.lists_dir, args.threads)
    print(f"[calib] val split: {multi_probs.shape[0]} clips x {multi_probs.shape[1]} categories "
          f"(method={args.method})")
    if args.method == "temperature":
        calibration = C.fit_per_class(multi_probs, multi_labels, CATEGORIES, split="val",
                                      n_bins=args.bins)
        payload = calibration.to_dict()
    else:
        payload = C.fit_per_class_isotonic(multi_probs, multi_labels, CATEGORIES, split="val",
                                           n_bins=args.bins)
        calibration = None
    payload["method"] = args.method

    per_class = payload["metrics"]["per_class"]
    unfitted = payload["unfitted"]
    mappings = payload.get("mappings", {})
    print(f"[calib] per-class {payload['method']} "
          f"(val: the split it was fitted on -> optimistic)")
    for category, entry in per_class.items():
        if entry["fitted"]:
            extra = (f" n_breaks={mappings[category]['metrics']['n_breaks']}"
                     if payload["method"] == "isotonic"
                     else f" T={entry['temperature']:.3f}")
            print(f"    {category:13s} n_pos={entry['n_pos']:3d}  "
                  f"ECE {entry['ece_before']:.4f} -> {entry['ece_after']:.4f}{extra}")
        else:
            print(f"    {category:13s} UNFITTED (n_pos={entry['n_pos']})  "
                  f"ECE {entry['ece']:.4f}  <- no label in val, stays uncalibrated")
    print(f"[calib] macro ECE over fitted heads "
          f"{payload['metrics']['macro_ece_before']:.4f} -> "
          f"{payload['metrics']['macro_ece_after']:.4f} | unfitted: {unfitted}")

    if args.test:
        test_probs, test_labels = test_clip_multilabel(args.ckpt, args.lists_dir, args.threads)
        mappings = payload.get("mappings", {})

        def _apply(p, category: str):
            if category in unfitted:
                return p
            if payload["method"] == "isotonic":
                return C.apply_isotonic(p, mappings[category]["mapping"])
            return calibration.probability(p, category)

        per_class: dict[str, dict] = {}
        for index, category in enumerate(CATEGORIES):
            p, y = test_probs[:, index], test_labels[:, index]
            calibrated = _apply(p, category)
            per_class[category] = {
                "n_pos": int(y.sum()),
                "temperature": (payload.get("temperatures", {}).get(category, 1.0)
                                if payload["method"] == "temperature" else None),
                "fitted": category not in unfitted,
                "ece_before": C.expected_calibration_error(p, y, args.bins),
                "ece_after": C.expected_calibration_error(calibrated, y, args.bins),
                "mean_confidence_before": float(p.mean()),
                "mean_confidence_after": float(calibrated.mean()),
                "positive_rate": float(y.mean()),
            }
        fitted = [c for c in CATEGORIES if c not in unfitted]
        before = float(np.mean([per_class[c]["ece_before"] for c in fitted]))
        after = float(np.mean([per_class[c]["ece_after"] for c in fitted]))
        payload["test"] = {"n_clips": int(test_probs.shape[0]), "macro_ece_before": before,
                           "macro_ece_after": after, "per_class": per_class,
                           "note": "temperatures fitted on val; test labels used only to "
                                   "measure, never to fit"}
        print(f"[calib] test split ({test_probs.shape[0]} clips) - the honest evaluation:")
        for category in CATEGORIES:
            entry = per_class[category]
            note = "" if entry["fitted"] else "  (unfitted: T=1)"
            print(f"    {category:13s} n_pos={entry['n_pos']:3d}  "
                  f"conf {entry['mean_confidence_before']:.3f} vs rate {entry['positive_rate']:.3f}"
                  f"  ECE {entry['ece_before']:.4f} -> {entry['ece_after']:.4f}{note}")
        print(f"[calib] test macro ECE {before:.4f} -> {after:.4f}")

        # Reference the alternative that already exists: the *single* global temperature from the
        # clip-level head. Per-class fitting is only worth its six parameters if it beats that.
        global_map = C.CalibrationMap.load(Path(args.ckpt).parent / "calibration.json")
        if global_map is not None:
            global_ece = [C.expected_calibration_error(global_map.probability(test_probs[:, i]),
                                                       test_labels[:, i], args.bins)
                          for i, c in enumerate(CATEGORIES) if c in fitted]
            macro_global = float(np.mean(global_ece))
            payload["test"]["global_temperature"] = global_map.temperature
            payload["test"]["global_temperature_macro_ece"] = macro_global
            options = {"per-class": after, "uncalibrated": before,
                       f"global T={global_map.temperature:.3f}": macro_global}
            best = min(options, key=options.get)
            payload["test"]["best_of"] = best
            print(f"[calib] test macro ECE - uncalibrated {before:.4f} | per-class {after:.4f} | "
                  f"one global T={global_map.temperature:.4f} {macro_global:.4f}")
            print(f"[calib] best of the three: {best}")
        else:
            print("[calib] no calibration.json - cannot compare against the global temperature")

    default_name = ("calibration_per_class.json" if payload["method"] == "temperature"
                    else f"calibration_per_class_{payload['method']}.json")
    out = Path(args.out) if args.out else Path(args.ckpt).parent / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calib] written -> {out}")
    return 0


def _reconstructed_test_scores(ckpt: str, top_k: int):
    """Rebuild clip-level test scores from the saved snippet curves (the training top-k rule).

    Returns ``(scores, labels)`` or ``None`` when the run has no ``predictions.npz`` yet.
    """
    pred = Path(ckpt).parent / "predictions.npz"
    if not pred.exists():
        print(f"[calib] no {pred.name} - score the run first (safewatch.eval.runner)")
        return None
    archive = np.load(pred)
    test_scores, test_labels = [], []
    for key in sorted(k for k in archive.files if k.startswith("scores::")):
        gt = archive.get(f"gt::{key[len('scores::'):]}")
        if gt is None:
            continue
        test_scores.append(C.clip_scores_from_snippet_curve(archive[key], top_k))
        test_labels.append(float(gt.max()))
    return np.asarray(test_scores), np.asarray(test_labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--lists-dir", default=None,
                        help="fallback when the checkpoint records none (default: data/lists)")
    parser.add_argument("--k", type=int, default=5, help="top-k rule for the test reconstruction")
    parser.add_argument("--bins", type=int, default=15, help="ECE / reliability-diagram bins")
    parser.add_argument("--test", action="store_true",
                        help="also report ECE on the saved test predictions (reconstructed score)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=None, help="default: <run dir>/calibration.json")
    parser.add_argument("--per-class", action="store_true",
                        help="P5: fit one temperature per category on the multi-label head and "
                             "report per-class ECE on val and test -> calibration_per_class.json")
    parser.add_argument("--method", choices=("temperature", "isotonic"), default="temperature",
                        help="calibration family: temperature scaling (default, one scalar) or "
                             "isotonic regression (non-parametric monotone map) - the untried rung "
                             "after the Session-14 temperature dead end")
    args = parser.parse_args()

    if args.per_class:
        return _run_per_class(args)

    scores, labels = val_clip_probabilities(args.ckpt, args.lists_dir, args.threads)
    print(f"[calib] val split: {scores.size} clips, positive rate {labels.mean():.3f}, "
          f"mean raw confidence {scores.mean():.3f} (method={args.method})")
    if args.method == "isotonic":
        payload = C.fit_isotonic(scores, labels, split="val", source="clip_logits",
                                 n_bins=args.bins)
        print(f"[calib] isotonic: {payload['metrics']['n_breaks']} breaks | ECE "
              f"{payload['metrics']['ece_before']:.4f} -> {payload['metrics']['ece_after']:.4f} | "
              f"NLL {payload['metrics']['nll_before']:.4f} -> "
              f"{payload['metrics']['nll_after']:.4f}")
        if args.test:
            reconstructed = _reconstructed_test_scores(args.ckpt, args.k)
            if reconstructed is not None:
                s, y = reconstructed
                calibrated = C.apply_isotonic(s, payload["mapping"])
                payload["test_reconstructed"] = {
                    "n_clips": int(s.size),
                    "note": "clip score rebuilt from the saved snippet curve with the training "
                            "top-k rule; isotonic map taken from val",
                    "ece_before": C.expected_calibration_error(s, y, args.bins),
                    "ece_after": C.expected_calibration_error(calibrated, y, args.bins),
                    "mean_confidence_before": float(s.mean()),
                    "mean_confidence_after": float(calibrated.mean()),
                    "positive_rate": float(y.mean()),
                }
                block = payload["test_reconstructed"]
                print(f"[calib] test (reconstructed, n={block['n_clips']}): "
                      f"ECE {block['ece_before']:.4f} -> {block['ece_after']:.4f} | "
                      f"mean confidence {s.mean():.3f} -> {calibrated.mean():.3f} "
                      f"vs positive rate {y.mean():.3f}")
        out = Path(args.out) if args.out else Path(args.ckpt).parent / "calibration_isotonic.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[calib] written -> {out}")
        return 0
    calibration = C.fit(scores, labels, split="val", source="clip_logits", n_bins=args.bins)
    payload = calibration.to_dict()
    payload["reliability_val"] = C.reliability_bins(calibration.probability(scores), labels,
                                                    args.bins)
    print(f"[calib] temperature = {calibration.temperature:.4f} | ECE "
          f"{calibration.metrics['ece_before']:.4f} -> {calibration.metrics['ece_after']:.4f} | "
          f"NLL {calibration.metrics['nll_before']:.4f} -> {calibration.metrics['nll_after']:.4f}")

    if args.test:
        reconstructed = _reconstructed_test_scores(args.ckpt, args.k)
        if reconstructed is not None:
            s, y = reconstructed
            calibrated = calibration.probability(s)
            payload["test_reconstructed"] = {
                "n_clips": int(s.size),
                "note": "clip score rebuilt from the saved snippet curve with the training top-k "
                        "rule; temperature taken from val",
                "ece_before": C.expected_calibration_error(s, y, args.bins),
                "ece_after": C.expected_calibration_error(calibrated, y, args.bins),
                "mean_confidence_before": float(s.mean()),
                "mean_confidence_after": float(calibrated.mean()),
                "positive_rate": float(y.mean()),
            }
            block = payload["test_reconstructed"]
            print(f"[calib] test (reconstructed, n={block['n_clips']}): "
                  f"ECE {block['ece_before']:.4f} -> {block['ece_after']:.4f} | mean confidence "
                  f"{s.mean():.3f} -> {calibrated.mean():.3f} vs positive rate {y.mean():.3f}")

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calib] written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
