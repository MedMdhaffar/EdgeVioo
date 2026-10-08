"""Confidence calibration for the alerts (subject: "estimer un niveau de confiance").

Why this is not cosmetic: the alert card shows a percentage and an operator acts on it. Raw sigmoid
outputs of a MIL model trained with a top-k objective are *ranking* scores, not probabilities - the
numbers below are measured on this project's runs, not assumed (`docs/journal.md`, Session 10).

Method: **temperature scaling** (Guo et al. 2017), one scalar fitted by minimising BCE on the
**validation split**, i.e. the split already used for model selection, so no test label is ever
read. Evidence reported next to it: ECE (expected calibration error), NLL, and the reliability
diagram bins.

Two honest limitations, repeated in the report template rather than hidden:
* the temperature is fitted at **clip** level (the val split has weak labels only) and then applied
  to snippet logits as well; that assumes both share one scale. Fitting on frames would need test
  labels, i.e. leakage - refused.
* ECE measured on the split the temperature was fitted on is optimistic; the test-split value is
  reported beside it, with the note that the temperature came from val.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

EPS = 1e-7


def _logit(probability: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


# --------------------------------------------------------------------------- #
# ECE / reliability diagram
# --------------------------------------------------------------------------- #
def reliability_bins(probabilities, labels, n_bins: int = 15) -> dict:
    """Equal-width bins over [0, 1]: mean confidence, empirical accuracy and count per bin.

    Empty bins are returned with ``count=0`` so the reliability diagram can be drawn without gaps.
    """
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if probs.size != y.size:
        raise ValueError(f"{probs.size} probabilities vs {y.size} labels")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    confidence = np.zeros(n_bins)
    accuracy = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = index == b
        count[b] = int(mask.sum())
        if count[b]:
            confidence[b] = probs[mask].mean()
            accuracy[b] = y[mask].mean()
    return {"bin_edges": edges.tolist(), "confidence": confidence.tolist(),
            "accuracy": accuracy.tolist(), "count": count.tolist()}


def expected_calibration_error(probabilities, labels, n_bins: int = 15) -> float:
    """|accuracy - confidence| averaged over bins, weighted by bin population (ECE)."""
    bins = reliability_bins(probabilities, labels, n_bins)
    count = np.asarray(bins["count"], dtype=np.float64)
    total = count.sum()
    if total == 0:
        return float("nan")
    gap = np.abs(np.asarray(bins["accuracy"]) - np.asarray(bins["confidence"]))
    return float((gap * count).sum() / total)


def binary_nll(probabilities, labels) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64).ravel(), EPS, 1 - EPS)
    y = np.asarray(labels, dtype=np.float64).ravel()
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())



# --------------------------------------------------------------------------- #
# temperature scaling
# --------------------------------------------------------------------------- #
def fit_temperature(clip_probabilities, labels, max_iter: int = 200,
                    init: float = 1.0) -> float:
    """One scalar T minimising BCE between ``sigmoid(logit(p) / T)`` and the labels.

    ``T > 1`` softens an over-confident model (the usual case for these heads), ``T < 1`` sharpens
    an under-confident one. Fitted with LBFGS on one parameter, deterministic given the inputs.
    """
    logits = torch.tensor(_logit(clip_probabilities), dtype=torch.float32)
    target = torch.tensor(np.asarray(labels, dtype=np.float64).ravel(), dtype=torch.float32)
    if logits.numel() != target.numel():
        raise ValueError(f"{logits.numel()} scores vs {target.numel()} labels")
    if torch.unique(target).numel() < 2:
        raise ValueError("temperature needs both classes in the fitting split")

    log_temperature = torch.tensor(float(np.log(init)), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits / torch.exp(log_temperature).clamp(1e-3, 1e3)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp(1e-3, 1e3).item())


@dataclass
class CalibrationMap:
    """A fitted temperature plus the evidence that it helps (saved as ``calibration.json``)."""

    temperature: float = 1.0
    fitted_on: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def probability(self, logits_or_probs, already_probability: bool = True) -> np.ndarray:
        """Calibrated probabilities from raw probabilities (default) or raw logits."""
        values = np.asarray(logits_or_probs, dtype=np.float64)
        logits = _logit(values) if already_probability else values
        return _sigmoid(logits / max(self.temperature, EPS))

    @property
    def is_identity(self) -> bool:
        return abs(self.temperature - 1.0) < 1e-6

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "fitted_on": self.fitted_on,
                "metrics": self.metrics}

    @classmethod
    def from_dict(cls, payload: dict) -> CalibrationMap:
        return cls(temperature=float(payload.get("temperature", 1.0)),
                   fitted_on=dict(payload.get("fitted_on", {})),
                   metrics=dict(payload.get("metrics", {})))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> CalibrationMap | None:
        """``None`` when the file is absent: callers must then say "uncalibrated" out loud."""
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))



def fit(scores, labels, split: str = "val", source: str = "clip_logits",
        n_bins: int = 15) -> CalibrationMap:
    """Fit the temperature and record before/after ECE + NLL as the justification."""
    probs = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    temperature = fit_temperature(probs, y)
    calibrated = CalibrationMap(temperature=temperature).probability(probs)
    metrics = {
        "ece_before": expected_calibration_error(probs, y, n_bins),
        "ece_after": expected_calibration_error(calibrated, y, n_bins),
        "nll_before": binary_nll(probs, y),
        "nll_after": binary_nll(calibrated, y),
        "mean_confidence_before": float(probs.mean()),
        "mean_confidence_after": float(calibrated.mean()),
        "positive_rate": float(y.mean()),
        "n_bins": n_bins,
    }
    fitted = {"split": split, "source": source, "n_samples": int(probs.size)}
    return CalibrationMap(temperature=temperature, fitted_on=fitted, metrics=metrics)


@dataclass
class PerClassCalibration:
    """One temperature per category, fitted on the weak clip labels of one split.

    Why this exists: the alert card names a *category* ("riot, 91 %"), so the number an operator
    acts on is the multi-label head, not the clip-level anomaly score that :class:`CalibrationMap`
    calibrates. A single temperature for the whole head cannot fix a head that is over-confident on
    one class and under-confident on another.

    Failure mode kept visible: a category with a single-class target in the fitting split cannot be
    fitted (``fit_temperature`` refuses it). The official val split contains **no** ``abuse`` clip
    at all, so that head stays at ``T=1`` and is listed in :attr:`unfitted` - an operator must be
    able to tell "abuse 12 %" (never calibrated) from a genuine probability.
    """

    maps: dict = field(default_factory=dict)
    unfitted: list = field(default_factory=list)
    fitted_on: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def temperature(self, category: str) -> float:
        entry = self.maps.get(category)
        return float(entry.temperature) if entry is not None else 1.0

    def probability(self, probabilities, category: str) -> np.ndarray:
        """Calibrate one category (identity when that head could not be fitted)."""
        entry = self.maps.get(category)
        if entry is None:
            return np.asarray(probabilities, dtype=np.float64)
        return entry.probability(probabilities)

    def to_dict(self) -> dict:
        return {"temperatures": {k: v.temperature for k, v in self.maps.items()},
                "unfitted": list(self.unfitted), "fitted_on": self.fitted_on,
                "metrics": self.metrics}

    @classmethod
    def from_dict(cls, payload: dict) -> PerClassCalibration:
        maps = {k: CalibrationMap(temperature=float(t))
                for k, t in dict(payload.get("temperatures", {})).items()}
        return cls(maps=maps, unfitted=list(payload.get("unfitted", [])),
                   fitted_on=dict(payload.get("fitted_on", {})),
                   metrics=dict(payload.get("metrics", {})))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> PerClassCalibration | None:
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


def fit_per_class(probabilities, labels, categories, split: str = "val",
                  n_bins: int = 15) -> PerClassCalibration:
    """Fit one temperature per category from ``(n_clips, n_categories)`` probabilities and labels.

    ``unfitted`` lists the heads the split cannot support (all-positive or all-negative target,
    which happens for a class absent from the split). Their evidence is still reported, with
    identity calibration, so the gap is visible rather than hidden.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    if probs.shape != truth.shape:
        raise ValueError(f"probabilities {probs.shape} vs labels {truth.shape}")
    if probs.shape[1] != len(categories):
        raise ValueError(f"{probs.shape[1]} columns vs {len(categories)} categories")

    maps: dict[str, CalibrationMap] = {}
    unfitted: list[str] = []
    per_class: dict[str, dict] = {}
    for index, category in enumerate(categories):
        p, y = probs[:, index], truth[:, index]
        entry: dict = {"n": int(p.size), "n_pos": int(y.sum())}
        try:
            temperature = fit_temperature(p, y)
        except ValueError:                     # single-class target: nothing to fit on
            unfitted.append(category)
            entry.update({"temperature": 1.0, "fitted": False,
                          "ece": expected_calibration_error(p, y, n_bins),
                          "nll": binary_nll(p, y)})
            per_class[category] = entry
            continue
        calibrated = CalibrationMap(temperature=temperature).probability(p)
        entry.update({
            "temperature": temperature, "fitted": True,
            "ece_before": expected_calibration_error(p, y, n_bins),
            "ece_after": expected_calibration_error(calibrated, y, n_bins),
            "nll_before": binary_nll(p, y), "nll_after": binary_nll(calibrated, y),
            "mean_confidence_before": float(p.mean()),
            "mean_confidence_after": float(calibrated.mean()),
            "positive_rate": float(y.mean()),
        })
        maps[category] = CalibrationMap(temperature=temperature)
        per_class[category] = entry

    fitted = [c for c in categories if c not in unfitted]
    macro = {
        "macro_ece_before": float(np.mean([per_class[c]["ece_before"] for c in fitted]))
        if fitted else float("nan"),
        "macro_ece_after": float(np.mean([per_class[c]["ece_after"] for c in fitted]))
        if fitted else float("nan"),
    }
    fitted_on = {"split": split, "n_samples": int(probs.shape[0]),
                 "n_categories": len(categories), "fitted_categories": fitted}
    return PerClassCalibration(maps=maps, unfitted=unfitted, fitted_on=fitted_on,
                               metrics={**macro, "per_class": per_class, "n_bins": n_bins})


def clip_scores_from_snippet_curve(scores, k: int = 5) -> float:
    """Clip-level score rebuilt from a saved snippet curve with the training objective's top-k rule.

    Used when only ``predictions.npz`` is available (the runner stores snippet curves, not clip
    logits). Documented as a reconstruction wherever it is reported.
    """
    values = np.asarray(scores).ravel()
    if values.size == 0:
        return float("nan")
    k_eff = max(1, min(k, values.size))
    return float(np.sort(values)[-k_eff:].mean())


# --------------------------------------------------------------------------- #
# isotonic regression (the non-parametric alternative to temperature scaling)
# --------------------------------------------------------------------------- #
#
# Why it exists: temperature scaling is a one-parameter *monotone* map. When it measurably fails to
# lower ECE (Session 14: it raised the category head's ECE on both runs), the next step in the
# standard ladder is the fully non-parametric monotone map - isotonic regression - which cannot
# correct a *non-monotone* miscalibration but can reshape the calibration curve arbitrarily within
# the monotone family. With ~150 val clips it is less forgiving of overfitting than temperature;
# the fitted step function is stored verbatim (f_.x, f_.y) so the applied mapping is exactly the
# one that was fitted, and the report says so.


def fit_isotonic(scores, labels, split: str = "val", source: str = "clip_logits",
                 n_bins: int = 15) -> dict:
    """Fit isotonic regression and record the same evidence block as :func:`fit`."""
    from sklearn.isotonic import IsotonicRegression

    probs = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if probs.size != y.size:
        raise ValueError(f"{probs.size} scores vs {y.size} labels")
    if len(np.unique(y)) < 2:
        raise ValueError("isotonic fit needs both classes in the fitting split")

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(probs, y)
    calibrated = iso.predict(probs)

    metrics = {
        "ece_before": expected_calibration_error(probs, y, n_bins),
        "ece_after": expected_calibration_error(calibrated, y, n_bins),
        "nll_before": binary_nll(probs, y),
        "nll_after": binary_nll(calibrated, y),
        "mean_confidence_before": float(probs.mean()),
        "mean_confidence_after": float(calibrated.mean()),
        "positive_rate": float(y.mean()),
        "n_bins": n_bins,
        "n_breaks": int(iso.f_.x.size),
    }
    return {
        "method": "isotonic",
        "mapping": {"x": [float(v) for v in iso.f_.x],
                    "y": [float(v) for v in iso.f_.y]},
        "fitted_on": {"split": split, "source": source, "n_samples": int(probs.size)},
        "metrics": metrics,
    }


def apply_isotonic(probabilities, mapping: dict) -> np.ndarray:
    """Apply a stored isotonic step mapping (``np.interp`` with clipped ends)."""
    x = np.asarray(mapping["x"], dtype=np.float64)
    y = np.asarray(mapping["y"], dtype=np.float64)
    return np.interp(np.asarray(probabilities, dtype=np.float64).ravel(), x, y,
                     left=float(y[0]), right=float(y[-1]))


def fit_per_class_isotonic(probabilities, labels, categories, split: str = "val",
                           n_bins: int = 15) -> dict:
    """One isotonic map per category; single-class targets stay unfitted (T=1 analogue: identity).

    Mirrors :func:`fit_per_class`'s contract: the fitting split may lack a whole class (the
    official val has zero ``abuse`` clips), and that gap is reported, not hidden.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    if probs.shape != truth.shape:
        raise ValueError(f"probabilities {probs.shape} vs labels {truth.shape}")
    if probs.shape[1] != len(categories):
        raise ValueError(f"{probs.shape[1]} columns vs {len(categories)} categories")

    maps: dict[str, dict] = {}
    unfitted: list[str] = []
    per_class: dict[str, dict] = {}
    for index, category in enumerate(categories):
        p, yv = probs[:, index], truth[:, index]
        entry: dict = {"n": int(p.size), "n_pos": int(yv.sum())}
        if len(np.unique(yv)) < 2:
            unfitted.append(category)
            entry.update({"fitted": False,
                          "ece": expected_calibration_error(p, yv, n_bins),
                          "nll": binary_nll(p, yv)})
            per_class[category] = entry
            continue
        entry.update({"fitted": True,
                      "ece_before": expected_calibration_error(p, yv, n_bins),
                      "nll_before": binary_nll(p, yv),
                      "mean_confidence_before": float(p.mean()),
                      "positive_rate": float(yv.mean())})
        maps[category] = fit_isotonic(p, yv, n_bins=n_bins)
        entry.update({"ece_after": maps[category]["metrics"]["ece_after"],
                      "nll_after": maps[category]["metrics"]["nll_after"],
                      "mean_confidence_after": maps[category]["metrics"]["mean_confidence_after"]})
        per_class[category] = entry
    fitted = [c for c in categories if c not in unfitted]
    macro = {
        "macro_ece_before": float(np.mean([per_class[c]["ece_before"] for c in fitted]))
        if fitted else float("nan"),
        "macro_ece_after": float(np.mean([per_class[c]["ece_after"] for c in fitted]))
        if fitted else float("nan"),
    }
    return {
        "method": "isotonic",
        "temperatures": {},            # not applicable; kept for loader compatibility
        "unfitted": unfitted,
        "fitted_on": {"split": split, "n_samples": int(probs.shape[0]),
                      "n_categories": len(categories), "fitted_categories": fitted},
        "metrics": {**macro, "per_class": per_class, "n_bins": n_bins},
        "mappings": maps,
    }


def _single_isotonic(p, yv, n_bins: int = 15) -> dict:
    return fit_isotonic(p, yv)
