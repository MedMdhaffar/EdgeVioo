"""Clip-level dataset for weakly-supervised MIL training on XD-Violence (HF mirror).

Each clip is one feature file of shape ``(T, 5 crops, D)`` (D = 768 for VideoSwin, 2048 for I3D).
Crops are reduced (mean of the 5, or centre only), the sequence is either cropped randomly
(training), uniformly sub-sampled (validation/test) or kept whole (``max_snippets=None``, used for
full-length inference), and short clips are zero-padded. A boolean mask marks real snippets so that
pooling and losses can ignore padding.

Category order is fixed by ``CATEGORIES`` and matches the order used in ``data/lists/*.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORIES = ("fighting", "shooting", "riot", "abuse", "car_accident", "explosion")
N_CATEGORIES = len(CATEGORIES)


def auto_pos_weight(rows: list[dict], mode: str = "auto",
                    max_weight: float = 100.0):
    """Per-category positive weights for the multi-label BCE, from the split's weak labels.

    ``mode="auto"`` gives ``n_negative / n_positive`` (the standard inverse-frequency weight) and
    ``mode="sqrt"`` its square root - a damped middle ground, because the raw ratio is brutal here:
    ``abuse`` scores 3 754 / 50 = 75 against 7-9 for the other classes, so the unweighted 6-class
    head is effectively trained on five classes (see ``docs/journal.md`` Session 10 addendum).

    Weights are capped at ``max_weight`` so a class with a handful of positives cannot blow up the
    loss. Returns a float32 tensor of length ``N_CATEGORIES``, ordered like :data:`CATEGORIES`.
    """
    import torch

    if mode not in ("auto", "sqrt"):
        raise ValueError(f"unknown pos-weight mode {mode!r}; expected 'auto' or 'sqrt'")
    counts = np.zeros(N_CATEGORIES, dtype=np.float64)
    for row in rows:
        for code in (row.get("labels") or "").split("|"):
            if code in CATEGORIES:
                counts[CATEGORIES.index(code)] += 1.0
    positives = np.maximum(counts, 1.0)
    negatives = np.maximum(len(rows) - counts, 0.0)
    weight = negatives / positives
    if mode == "sqrt":
        weight = np.sqrt(weight)
    return torch.tensor(np.minimum(weight, max_weight), dtype=torch.float32)


def auto_row_weight(rows: list[dict], max_weight: float = 10.0) -> "np.ndarray":
    """Per-row sampling weights for ``WeightedRandomSampler``: rare-class clips get *exposure*.

    The remedy for the data starvation diagnosed in ``docs/journal.md`` (Session 10 addendum /
    Session 11): ``abuse`` has 50 positive clips out of 3 804, so a random loader shows each of
    them roughly once per epoch while the common classes are shown ~9 times. Loss re-weighting
    (``auto_pos_weight``) was measured and failed as a ranking remedy, but it does *not* change
    how often a clip is seen. Here a row's weight is the inverse-frequency ratio of its rarer
    category, capped at ``max_weight``; a normal or single-common-class clip keeps weight 1, so
    the majority distribution is unchanged and only the rare classes are oversampled (their
    random crops still differ between appearances, so this is exposure, not memorisation).
    """
    counts = np.zeros(N_CATEGORIES, dtype=np.float64)
    for row in rows:
        for code in (row.get("labels") or "").split("|"):
            if code in CATEGORIES:
                counts[CATEGORIES.index(code)] += 1.0
    weights = np.ones(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        rarest = 1.0
        for code in (row.get("labels") or "").split("|"):
            if code in CATEGORIES:
                positives = max(1.0, counts[CATEGORIES.index(code)])
                negatives = max(0.0, len(rows) - positives)
                rarest = max(rarest, negatives / positives)
        weights[index] = min(rarest, max_weight)
    return weights


def load_rows(csv_path: str | Path) -> list[dict]:
    """Read one of the split CSVs produced by ``scripts/build_lists.py``."""
    import csv

    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class ClipDataset(Dataset):
    """One item = one clip: snippet features + mask + binary and multi-label targets."""

    def __init__(self, csv_path: str | Path, max_snippets: int | None = 200,
                 crop: str = "mean", train: bool = True, seed: int = 0) -> None:
        self.rows = load_rows(csv_path)
        self.max_snippets = max_snippets
        self.crop = crop
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: str) -> np.ndarray:
        if not path.endswith(".npy"):
            # official layout: one file per crop, <prefix>__0.npy .. <prefix>__4.npy  -> (T, 5, D)
            crops = [np.asarray(np.load(f"{path}__{c}.npy", mmap_mode="r"), dtype=np.float32)
                     for c in range(5)]
            arr = np.stack(crops, axis=1)
        else:
            arr = np.load(path, mmap_mode="r")
        if arr.ndim == 3:  # (T, crops, D)
            arr = arr.mean(axis=1) if self.crop == "mean" else arr[:, 0, :]
        return np.asarray(arr, dtype=np.float32)

    def _sequence(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crop / sub-sample / pad to a fixed length; returns (features, mask)."""
        t = feats.shape[0]
        if t == 0:
            raise ValueError("empty feature array")
        if self.max_snippets is None:  # full-length inference
            return feats, np.ones(t, dtype=bool)
        n = self.max_snippets
        if t >= n:
            if self.train:  # random crop: different part of the clip every epoch
                start = int(self.rng.integers(0, t - n + 1))
                return feats[start:start + n], np.ones(n, dtype=bool)
            idx = np.linspace(0, t - 1, n).round().astype(int)  # deterministic coverage
            return feats[idx], np.ones(n, dtype=bool)
        pad = np.zeros((n - t, feats.shape[1]), dtype=feats.dtype)
        mask = np.concatenate([np.ones(t, dtype=bool), np.zeros(n - t, dtype=bool)])
        return np.concatenate([feats, pad], axis=0), mask

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        feats, mask = self._sequence(self._load(row.get("path") or row.get("prefix")))
        multi = np.zeros(N_CATEGORIES, dtype=np.float32)
        for code in (row["labels"] or "").split("|"):
            if code in CATEGORIES:
                multi[CATEGORIES.index(code)] = 1.0
        return {
            "features": torch.from_numpy(feats),
            "mask": torch.from_numpy(mask),
            "binary": torch.tensor(float(row["binary"]), dtype=torch.float32),
            "multi": torch.from_numpy(multi),
            "clip_id": row["clip_id"],
            "n_snippets": int(row["T"]),
        }