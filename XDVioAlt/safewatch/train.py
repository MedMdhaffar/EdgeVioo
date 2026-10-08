#!/usr/bin/env python
"""Unimodal MIL training on XD-Violence features (weak supervision): visual or audio.

Fusion (both modalities) lives in ``safewatch.train_fusion``; this CLI is the P1/P2 baseline.

Example:
    .venv/bin/python -m safewatch.train --tag p1_visual --epochs 30

Artifacts (there is no Git history by design - the run folder *is* the record):
    runs/<date>_<tag>/config.json    exact configuration used
    runs/<date>_<tag>/history.csv    per-epoch train loss + validation AP
    runs/<date>_<tag>/metrics.json   best-epoch metrics
    runs/<date>_<tag>/ckpt_best.pt   best checkpoint (selected on the validation split only)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from safewatch.data.multimodal import AUDIO_DIM, MultimodalClipDataset
from safewatch.eval import metrics as M
from safewatch.losses.mil import total_loss
from safewatch.models.mil import MILModel

IN_DIMS = {"swin_rgb": 768, "i3d_rgb": 2048, "c3d_rgb": 4096, "i3d_official": 1024}


@dataclass
class TrainConfig:
    """Everything that defines a run; serialised to config.json for reproducibility."""

    tag: str = "p1_visual"
    feature_set: str = "swin_rgb"
    modality: str = "visual"           # visual | audio (cross-modal fusion arrives in P3)
    audio_dir: str = "data/features/mel"
    audio_stats: str = "data/features/audio_stats.json"
    audio_norm: str = "none"          # "clip" = instance-normalise audio per clip (calibration fix)
    audio_grid: str = "patch"         # "patch" = 0.96 s mel hop; "snippet" = official VGGish
    audio_dim: int = AUDIO_DIM        # width of the audio features (128 for official VGGish)
    lists_dir: str = "data/lists"
    runs_dir: str = "runs"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_snippets: int = 200
    crop: str = "mean"
    k: int = 5
    w_attn: float = 0.5
    w_multi: float = 0.3
    hidden: int = 512
    emb: int = 128
    dropout: float = 0.6
    workers: int = 4
    seed: int = 42
    threads: int = 8
    lr_milestones: tuple = (10,)
    lr_gamma: float = 0.1

    @property
    def feature_key(self) -> str:
        """Which batch key feeds the model."""
        return "audio" if self.modality == "audio" else "visual"

    @property
    def in_dim(self) -> int:
        if self.modality == "audio":
            return self.audio_dim   # 132 = log-mel patch stats, 128 = official VGGish
        return IN_DIMS.get(self.feature_set, 768)

    def to_dict(self) -> dict:
        """Config as written to disk: dataclass fields **plus** the derived dimensions.

        ``in_dim``/``feature_key`` are properties, so a bare ``asdict`` loses them and downstream
        scoring has to guess the input dimension back from the feature-set name - which silently
        breaks every feature set missing from the scorer's lookup table (i3d_official, 1024-d).
        """
        return {**asdict(self), "in_dim": self.in_dim, "feature_key": self.feature_key}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_dataset(cfg: TrainConfig, csv_path: Path, train: bool) -> MultimodalClipDataset:
    """Feature dataset for ``cfg.modality`` - always the multimodal class, for **both** modalities.

    Every modality goes through :class:`MultimodalClipDataset` so the batch keys are exactly
    ``cfg.feature_key`` (``"visual"`` / ``"audio"``). Serving visual samples from a plain
    ``ClipDataset`` (whose key is ``"features"``) made every visual run die on its first batch with
    ``KeyError: 'visual'`` (see ``data/logs/p1_official.log``); ``tests/test_pipeline.py`` now locks
    the contract down.
    """
    if cfg.modality == "both":
        raise ValueError("modality='both' is a fusion setting - use safewatch.train_fusion")
    return MultimodalClipDataset(csv_path, cfg.audio_dir, modality=cfg.modality,
                                 audio_stats=cfg.audio_stats, audio_norm=cfg.audio_norm,
                                 audio_grid=cfg.audio_grid,
                                 max_snippets=cfg.max_snippets,
                                 crop=cfg.crop, train=train, seed=cfg.seed)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = _make_dataset(cfg, Path(cfg.lists_dir) / "train.csv", True)
    val_ds = _make_dataset(cfg, Path(cfg.lists_dir) / "val.csv", False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers)
    return train_loader, val_loader


@torch.no_grad()
def validate(model: MILModel, loader: DataLoader, device: torch.device,
             feature_key: str = "visual") -> dict:
    """Validate with **clip-level** scores only (weak labels; the test GT stays untouched)."""
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        out = model(batch[feature_key].to(device), batch["mask"].to(device))
        scores.append(torch.sigmoid(out["clip_logits"]).cpu().numpy())
        labels.append(batch["binary"].numpy())
    y = np.concatenate(labels)
    s = np.concatenate(scores)
    return {
        "val_pr_auc": M.pr_auc(y, s),
        "val_ap": M.average_precision(y, s),
        "val_n": int(y.size),
        "val_prevalence": float(y.mean()),
    }


def train(cfg: TrainConfig) -> dict:
    """Run training; writes the run folder and returns the best-epoch metrics."""
    torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg.runs_dir) / f"{date.today().isoformat()}_{cfg.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    train_loader, val_loader = build_loaders(cfg)
    model = MILModel(in_dim=cfg.in_dim, hidden=cfg.hidden, emb=cfg.emb, dropout=cfg.dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma)

    print(f"[run] {run_dir}")
    print(f"[cfg] modality={cfg.modality} feature_set={cfg.feature_set} in_dim={cfg.in_dim} "
          f"epochs={cfg.epochs} batch={cfg.batch_size} lr={cfg.lr} k={cfg.k}")
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"device={device} threads={torch.get_num_threads()}")

    best: dict = {"val_ap": -1.0, "val_pr_auc": float("nan"), "epoch": -1}
    history_path = run_dir / "history.csv"
    history_path.write_text(
        "epoch,train_loss,loss_mil,loss_attn,loss_multi,val_pr_auc,val_ap,seconds\n",
        encoding="utf-8")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started = time.perf_counter()
        running: dict[str, float] = {}
        n_steps = 0
        for batch in train_loader:
            targets = {key: batch[key].to(device) for key in ("mask", "binary", "multi")}
            out = model(batch[cfg.feature_key].to(device), batch["mask"].to(device))
            loss, parts = total_loss(out, targets, k=cfg.k, w_attn=cfg.w_attn,
                                     w_multi=cfg.w_multi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            n_steps += 1
        scheduler.step()

        means = {key: value / max(1, n_steps) for key, value in running.items()}
        val = validate(model, val_loader, device, cfg.feature_key)
        seconds = time.perf_counter() - started
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch},{means['loss']:.4f},{means['loss_mil']:.4f},"
                     f"{means['loss_attn']:.4f},{means['loss_multi']:.4f},"
                     f"{val['val_pr_auc']:.4f},{val['val_ap']:.4f},{seconds:.1f}\n")

        improved = val["val_ap"] > best["val_ap"]
        if improved:
            best = {**val, "epoch": epoch}
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch,
                        "val": val}, run_dir / "ckpt_best.pt")
        print(f"[epoch {epoch:3d}/{cfg.epochs}] loss={means['loss']:.4f} "
              f"(mil={means['loss_mil']:.4f}) val_pr_auc={val['val_pr_auc']:.4f} "
              f"val_ap={val['val_ap']:.4f}{' *best*' if improved else ''} [{seconds:.1f}s]")

    (run_dir / "metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"[done] best epoch {best['epoch']}: val_ap={best['val_ap']:.4f} "
          f"val_pr_auc={best['val_pr_auc']:.4f}")
    print(f"[done] artifacts -> {run_dir}")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(TrainConfig()).items():
        cli = f"--{name.replace('_', '-')}"
        if isinstance(default, tuple):
            parser.add_argument(cli, type=int, nargs="+", default=list(default))
        else:
            parser.add_argument(cli, type=type(default), default=default)
    cfg = TrainConfig(**vars(parser.parse_args()))
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
