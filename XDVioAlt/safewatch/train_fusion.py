#!/usr/bin/env python
"""P3 - fusion training (early / late / cross / adaptive) on the clips that have audio features.

Usage:
    .venv/bin/python -m safewatch.train_fusion --fusion-mode adaptive --tag p3_adaptive

Writes the usual run folder (config.json, history.csv, metrics.json, ckpt_best.pt), so results are
directly comparable with the unimodal runs produced by ``safewatch.train``.
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

from safewatch.data.dataset import CATEGORIES, N_CATEGORIES, auto_pos_weight, auto_row_weight
from safewatch.data.multimodal import (
    DEFAULT_QUALITY_DIR,
    DEFAULT_QUALITY_STATS,
    MultimodalClipDataset,
    audit_audio_rows,
    report_audio_preflight,
)
from safewatch.data.reliability import RELIABILITY_DIM, drop_modality
from safewatch.eval import metrics as M
from safewatch.eval.runner import STRIDE_FRAMES
from safewatch.losses.mil import total_loss
from safewatch.models.fusion import FusionModel


@dataclass
class FusionConfig:
    tag: str = "p3_fusion"
    fusion_mode: str = "adaptive"
    lists_dir: str = "data/lists"
    runs_dir: str = "runs"
    visual_dim: int = 768
    audio_dim: int = 132            # log-mel patch statistics; recorded so the scorer can rebuild
    feature_set: str = "swin_rgb"   # grid of the visual stream -> which frames/snippet to score
    audio_dir: str = "data/features/mel"
    audio_stats: str = "data/features/audio_stats.json"
    quality_dir: str = DEFAULT_QUALITY_DIR
    quality_stats: str = DEFAULT_QUALITY_STATS
    audio_norm: str = "none"        # "clip" = instance-normalise audio per clip (calibration fix)
    audio_grid: str = "patch"       # "patch" = our 0.96 s mel hop; "snippet" = official VGGish rows
    n_reliability: int = RELIABILITY_DIM   # gate input width (reliability channels); 0 = blind gate
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    max_snippets: int = 200
    k: int = 5
    w_attn: float = 0.5
    w_multi: float = 0.3
    emb: int = 128
    hidden: int = 512
    dropout: float = 0.6
    heads: int = 4
    workers: int = 4
    threads: int = 8
    seed: int = 42
    lr_milestones: tuple = (10,)
    lr_gamma: float = 0.1
    modality_dropout: float = 0.15  # zero one modality (and flag it in the reliability vector)
    multi_pos_weight: str = "none"  # "none" | "auto" (n_neg/n_pos) | "sqrt" (damped) per category
    gate_source: str = "embeddings"  # adaptive gate input: "embeddings" | "reliability"
    class_over_sample: str = "none"  # "none" | "auto": WeightedRandomSampler on rare-class clips
    max_row_weight: float = 10.0     # cap for the per-row exposure weight (class_over_sample)
    abuse_hard_negative_weight: float = 1.0  # extra abuse-head penalty on fighting/shooting negatives



def _dataset(cfg: FusionConfig, csv_path: Path, train: bool) -> MultimodalClipDataset:
    return MultimodalClipDataset(csv_path, cfg.audio_dir, modality="both",
                                 audio_stats=cfg.audio_stats, audio_norm=cfg.audio_norm,
                                 audio_grid=cfg.audio_grid,
                                 stride_frames=STRIDE_FRAMES.get(cfg.feature_set),
                                 quality_dir=cfg.quality_dir,
                                 quality_stats=cfg.quality_stats, max_snippets=cfg.max_snippets,
                                 crop="mean", train=train, seed=cfg.seed)


def _warn_val_blind_classes(rows: list[dict]) -> None:
    """Warn when the validation split cannot see a category at all (selection is blind to it).

    Measured on ``data/lists_official/val.csv`` (150 clips): every category appears except
    ``abuse``. Without this warning the missing class is invisible: ``val_multi_ap['abuse']`` is
    simply NaN while the checkpoint keeps being chosen on the binary score, so a class-reweighting
    run looks uneventful in the history even when its target class starts firing.
    """
    counts = {category: 0 for category in CATEGORIES}
    for row in rows:
        for code in (row.get("labels") or "").split("|"):
            if code in counts:
                counts[code] += 1
    missing = [category for category, n in counts.items() if n == 0]
    if missing:
        print(f"[warn] val split ({len(rows)} clips) has no {', '.join(missing)} clip(s): "
              f"model selection is blind to {'them' if len(missing) > 1 else 'it'}")


def build_loaders(cfg: FusionConfig) -> tuple[DataLoader, DataLoader]:
    from torch.utils.data import WeightedRandomSampler

    train_ds = _dataset(cfg, Path(cfg.lists_dir) / "train.csv", True)
    val_ds = _dataset(cfg, Path(cfg.lists_dir) / "val.csv", False)
    stride = STRIDE_FRAMES.get(cfg.feature_set, 64.0)
    for label, dataset in (("train", train_ds), ("validation", val_ds)):
        report_audio_preflight(
            audit_audio_rows(dataset.rows, cfg.audio_dir, cfg.audio_grid, cfg.audio_dim, stride,
                             quality_dir=cfg.quality_dir),
            label)
    if cfg.class_over_sample == "auto":
        # Rare-class exposure: abuse clips (50 / 3 804) are sampled ~10x more often than normal
        # clips. This is NOT the Session-11 loss re-weighting (which trades ranking for recall):
        # the loss is untouched, the clip simply gets more looks, each with a fresh random crop.
        row_weights = auto_row_weight(train_ds.rows, max_weight=cfg.max_row_weight)
        sampler = WeightedRandomSampler(row_weights, num_samples=len(train_ds),
                                        replacement=True,
                                        generator=torch.Generator().manual_seed(cfg.seed))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                                  num_workers=cfg.workers)
        print(f"[data] class_over_sample=auto: "
              f"{int((row_weights > 1.0).sum())} / {len(row_weights)} rows boosted "
              f"(max weight {row_weights.max():.1f}, seed {cfg.seed})")
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers)
    return train_loader, val_loader


def _reliability(batch: dict, device: torch.device) -> torch.Tensor | None:
    """Reliability tensor on the device, or ``None`` when the gate has no reliability input.

    Keeping this in one place matters: a checkpoint trained with ``n_reliability=0`` (the four
    2026-09-24 runs) must still load and score, and a model trained with the channel vector must
    never be fed ``None`` (its gate Linear layer would change width silently).
    """
    reliability = batch.get("reliability")
    return reliability.to(device) if reliability is not None else None


def _class_note(ap: float | None, category: str = "abuse") -> str:
    """One-word epoch line note for a rare category that the val split may not contain at all.

    The official val split has **no** ``abuse`` clip (measured: car_accident 21, fighting 18,
    riot 17, shooting 16, explosion 10, abuse 0), so its AP is NaN and model selection is blind to
    it. Printing ``n/a`` instead of ``nan`` keeps that visible instead of looking like a bug.
    """
    if ap is None or (isinstance(ap, float) and np.isnan(ap)):
        return f"val_{category}=n/a"
    return f"val_{category}_ap={ap:.4f}"


@torch.no_grad()
def validate(model: FusionModel, loader: DataLoader, device: torch.device) -> dict:
    """Clip-level validation (weak labels only; the test ground truth stays untouched).

    Reports the binary AP *and* the six per-category APs. The per-category part is what makes a
    class-reweighting experiment readable: the binary head can look unchanged while the category
    head starts firing for the first time (``abuse`` went from 0 TP to a live head this way).
    """
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    multi_scores: list[np.ndarray] = []
    multi_labels: list[np.ndarray] = []
    for batch in loader:
        out = model(batch["visual"].to(device), batch["audio"].to(device),
                    batch["mask"].to(device), _reliability(batch, device))
        scores.append(torch.sigmoid(out["clip_logits"]).cpu().numpy())
        labels.append(batch["binary"].numpy())
        multi_scores.append(torch.sigmoid(out["multi_logits"]).cpu().numpy())
        multi_labels.append(batch["multi"].numpy())
    y, s = np.concatenate(labels), np.concatenate(scores)
    my, ms = np.concatenate(multi_labels), np.concatenate(multi_scores)
    per_class = {category: M.average_precision(my[:, i], ms[:, i])
                 for i, category in enumerate(CATEGORIES)}
    return {"val_pr_auc": M.pr_auc(y, s), "val_ap": M.average_precision(y, s),
            "val_n": int(y.size), "val_prevalence": float(y.mean()),
            "val_multi_ap": per_class}


def train(cfg: FusionConfig) -> dict:
    torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg.runs_dir) / f"{date.today().isoformat()}_{cfg.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_loader, val_loader = build_loaders(cfg)
    if cfg.fusion_mode == "adaptive" and cfg.gate_source == "reliability" \
            and cfg.n_reliability <= 0:
        raise SystemExit("[cfg] gate_source=reliability needs n_reliability > 0 "
                         "(the gate must see the reliability channels)")
    model = FusionModel(visual_dim=cfg.visual_dim, audio_dim=cfg.audio_dim, emb=cfg.emb,
                        hidden=cfg.hidden, n_categories=N_CATEGORIES, dropout=cfg.dropout,
                        fusion_mode=cfg.fusion_mode, heads=cfg.heads,
                        n_reliability=cfg.n_reliability,
                        gate_source=cfg.gate_source).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(cfg.lr_milestones),
                                                     cfg.lr_gamma)

    pos_weight = None
    if cfg.multi_pos_weight != "none":
        pos_weight = auto_pos_weight(train_loader.dataset.rows, cfg.multi_pos_weight).to(device)
        print("[cfg] multi pos_weight (" + cfg.multi_pos_weight + "): "
              + ", ".join(f"{c}={float(w):.1f}" for c, w in
                          zip(CATEGORIES, pos_weight.cpu().numpy(), strict=True)))
    _warn_val_blind_classes(val_loader.dataset.rows)

    print(f"[run] {run_dir}")
    print(f"[cfg] fusion={cfg.fusion_mode} params={sum(p.numel() for p in model.parameters())} "
          f"reliability={cfg.n_reliability}d dropout={cfg.modality_dropout} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} device={device}")

    best: dict = {"val_ap": -1.0, "epoch": -1}
    history = run_dir / "history.csv"
    history.write_text("epoch,train_loss,val_pr_auc,val_ap,seconds\n", encoding="utf-8")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started = time.perf_counter()
        running, steps = 0.0, 0
        for batch in train_loader:
            visual, audio = batch["visual"].to(device), batch["audio"].to(device)
            mask = batch["mask"].to(device)
            reliability = _reliability(batch, device)
            if cfg.modality_dropout > 0 and random.random() < cfg.modality_dropout:
                # one modality is zeroed *and* flagged in the reliability vector - exactly the
                # substitution --drop-modality uses at scoring time (safewatch/data/reliability.py)
                visual, audio, reliability = drop_modality(
                    visual, audio, reliability, random.choice(("audio", "visual")))
            out = model(visual, audio, mask, reliability)
            targets = {key: batch[key].to(device) for key in ("mask", "binary", "multi")}
            loss, _parts = total_loss(out, targets, k=cfg.k, w_attn=cfg.w_attn,
                                      w_multi=cfg.w_multi, multi_pos_weight=pos_weight,
                                      abuse_hard_negative_weight=cfg.abuse_hard_negative_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        scheduler.step()

        val = validate(model, val_loader, device)
        seconds = time.perf_counter() - started
        with history.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch},{running / max(1, steps):.4f},{val['val_pr_auc']:.4f},"
                     f"{val['val_ap']:.4f},{seconds:.1f}\n")
        improved = val["val_ap"] > best["val_ap"]
        if improved:
            best = {**val, "epoch": epoch}
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch,
                        "val": val}, run_dir / "ckpt_best.pt")
        print(f"[epoch {epoch:3d}/{cfg.epochs}] loss={running / max(1, steps):.4f} "
              f"val_ap={val['val_ap']:.4f} "
              f"{_class_note(val['val_multi_ap'].get('abuse'))}"
              f"{' *best*' if improved else ''} [{seconds:.1f}s]")

    # Also keep the last epoch. Model selection watches the *binary* val AP, and the official val
    # split contains no ``abuse`` clip at all (see _warn_val_blind_classes), so the checkpoint that
    # ranks best on the binary task is not necessarily the one whose category head has learned the
    # rare class. Class-reweighting experiments must be able to compare both.
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": cfg.epochs,
                "val": val}, run_dir / "ckpt_last.pt")
    (run_dir / "metrics.json").write_text(json.dumps({**best, "last": val}, indent=2),
                                          encoding="utf-8")
    print(f"[done] best epoch {best['epoch']} val_ap={best['val_ap']:.4f} "
          f"| last val_ap={val['val_ap']:.4f} -> {run_dir}")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(FusionConfig()).items():
        cli = f"--{name.replace('_', '-')}"
        if isinstance(default, tuple):
            parser.add_argument(cli, type=int, nargs="+", default=list(default))
        else:
            parser.add_argument(cli, type=type(default), default=default)
    train(FusionConfig(**vars(parser.parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
