#!/usr/bin/env python3
"""Fine-tune the compatible checkpoint's snippet head from reviewed UCA time intervals.

Only explicitly reviewed violence intervals (positive) and explicitly reviewed normal intervals
(negative) supervise snippets. Unknown/blank/ignored time is masked out. The fixed XD validation
checkpoint is the initialization; UCA train fits and UCA val selects. UCA test is not loaded.

This script intentionally refuses to start until the review table has reviewed positive and normal
snippets in both training and validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"fighting", "shooting", "riot", "abuse", "car_accident", "explosion"}
VALID_LABELS = CLASSES | {"normal", "ignore"}
STRIDE_FRAMES = 16
FPS = 24.0
WINDOW = 200


def table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def feature_index(path: Path) -> dict[str, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["clip_id"]): item for item in manifest["videos"]}


def read_reviewed(rows: list[dict[str, str]], split: str) -> dict[str, list[tuple[float, float, int]]]:
    result: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for row in rows:
        if row["split"] != split:
            continue
        label = (row.get("review_label") or "").strip().lower()
        if not label or label == "ignore":
            continue
        if label not in VALID_LABELS:
            raise ValueError(f"invalid review_label {label!r} for {row['video_name']}")
        if label == "normal":
            target = 0
        else:
            target = 1
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if start < 0 or end <= start:
            raise ValueError(f"invalid reviewed interval in {row['video_name']}: {start}-{end}")
        clip_id = Path(row["video_name"]).stem
        result[clip_id].append((start, end, target))
    return result


def make_targets(n: int, intervals: list[tuple[float, float, int]], clip_id: str) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros(n, dtype=np.float32)
    known = np.zeros(n, dtype=bool)
    pos = np.zeros(n, dtype=bool)
    neg = np.zeros(n, dtype=bool)
    starts = np.arange(n, dtype=np.float32) * (STRIDE_FRAMES / FPS)
    ends = starts + STRIDE_FRAMES / FPS
    for begin, finish, value in intervals:
        overlaps = (starts < finish) & (ends > begin)
        if value:
            pos |= overlaps
        else:
            neg |= overlaps
    conflict = pos & neg
    if conflict.any():
        raise ValueError(f"conflicting positive/normal reviewed intervals in {clip_id} at {int(conflict.sum())} snippets")
    known = pos | neg
    target[pos] = 1.0
    return target, known


class ReviewedWindowDataset(Dataset):
    def __init__(self, split: str, intervals: dict[str, list[tuple[float, float, int]]],
                 features: dict[str, dict], audio_stats_path: Path, quality_stats_path: Path,
                 window: int = WINDOW):
        from safewatch.data.multimodal import clip_normalise, load_audio_stats, patches_per_snippet, pool_audio
        from safewatch.data.reliability import load_quality_stats, reliability_channels

        self.split, self.window = split, window
        self.audio_stats = load_audio_stats(audio_stats_path)
        self.quality_stats = load_quality_stats(quality_stats_path)
        self.pps = patches_per_snippet(STRIDE_FRAMES)
        self._pool_audio = pool_audio
        self._clip_normalise = clip_normalise
        self._reliability_channels = reliability_channels
        self.items: list[tuple[str, int]] = []
        self.labels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.features = features
        self.intervals = intervals
        for clip_id, events in intervals.items():
            meta = features.get(clip_id)
            if not meta or meta["split"] != split:
                raise ValueError(f"missing {split} feature record for reviewed clip {clip_id}")
            visual = np.load(ROOT / meta["visual_path"], mmap_mode="r", allow_pickle=False)
            target, known = make_targets(len(visual), events, clip_id)
            if not known.any():
                continue
            self.labels[clip_id] = (target, known)
            starts = list(range(0, len(visual), window))
            if len(visual) > window and starts and starts[-1] + window < len(visual):
                starts.append(len(visual) - window)
            for start in starts:
                if known[start:min(len(known), start + window)].any():
                    self.items.append((clip_id, start))
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, clip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if clip_id in self._arrays:
            return self._arrays[clip_id]
        meta = self.features[clip_id]
        visual = np.asarray(np.load(ROOT / meta["visual_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        raw_audio = np.asarray(np.load(ROOT / meta["audio_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        raw_quality = np.asarray(np.load(ROOT / meta["quality_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        n = len(visual)
        audio = self._pool_audio(raw_audio, n, self.pps)
        if self.audio_stats is not None and audio.shape[1] == self.audio_stats[0].shape[0]:
            audio = (audio - self.audio_stats[0]) / np.maximum(self.audio_stats[1], 1e-6)
        audio = self._clip_normalise(audio)
        reliability = self._reliability_channels(raw_quality, visual, n, self.quality_stats, self.pps)
        values = (visual, audio, reliability)
        self._arrays[clip_id] = values
        return values

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        clip_id, start = self.items[index]
        visual, audio, reliability = self._load(clip_id)
        end = min(len(visual), start + self.window)
        size = end - start
        v = np.zeros((self.window, visual.shape[1]), dtype=np.float32)
        a = np.zeros((self.window, audio.shape[1]), dtype=np.float32)
        r = np.zeros((self.window, reliability.shape[1]), dtype=np.float32)
        y = np.zeros(self.window, dtype=np.float32)
        known = np.zeros(self.window, dtype=bool)
        target, label_mask = self.labels[clip_id]
        v[:size], a[:size], r[:size] = visual[start:end], audio[start:end], reliability[start:end]
        y[:size], known[:size] = target[start:end], label_mask[start:end]
        return {"visual": torch.from_numpy(v), "audio": torch.from_numpy(a),
                "reliability": torch.from_numpy(r), "mask": torch.arange(self.window) < size,
                "known": torch.from_numpy(known), "target": torch.from_numpy(y), "clip_id": clip_id}


def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    ys, scores, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            visual, audio = batch["visual"].to(device), batch["audio"].to(device)
            mask, known = batch["mask"].to(device), batch["known"].to(device)
            reliability = batch["reliability"].to(device)
            target = batch["target"].to(device)
            logits = model(visual, audio, mask, reliability)["snippet_logits"]
            if known.any():
                losses.append(float(F.binary_cross_entropy_with_logits(logits[known], target[known])))
                ys.append(target[known].cpu().numpy())
                scores.append(torch.sigmoid(logits[known]).cpu().numpy())
    if not ys:
        return np.zeros(0), np.zeros(0), float("nan")
    return np.concatenate(ys), np.concatenate(scores), float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-table", default="data/annotations/uca_intervals_review.csv")
    parser.add_argument("--manifest", default="data/features/uca_feature_manifest.json")
    parser.add_argument("--base-checkpoint", default="runs/2026-10-08_binary_minirgb_sourceholdout_s43/ckpt_best.pt")
    parser.add_argument("--run-tag", default="uca_interval_head_finetune")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    review_path, manifest_path, base_path = ROOT / args.review_table, ROOT / args.manifest, ROOT / args.base_checkpoint
    if not review_path.is_file() or not manifest_path.is_file() or not base_path.is_file():
        raise SystemExit("Review table, feature manifest, or base checkpoint is missing")
    rows, features = table(review_path), feature_index(manifest_path)
    train_intervals, val_intervals = read_reviewed(rows, "train"), read_reviewed(rows, "val")
    if not train_intervals:
        raise SystemExit("No reviewed UCA train intervals. Review training rows before training.")
    if not val_intervals:
        raise SystemExit("No reviewed UCA validation intervals. Review validation rows before model selection.")

    ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_cfg = dict(ckpt.get("config", {}))
    from safewatch.eval.runner import load_model
    from safewatch.eval.metrics import average_precision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg, kind = load_model(base_path, device)
    if kind != "fusion" or not hasattr(model, "snippet_head"):
        raise SystemExit("Base checkpoint must be a fusion model with snippet_logits")
    expected_dim = int(base_cfg.get("visual_dim", 0))
    if expected_dim != 1024 or int(base_cfg.get("audio_dim", 0)) != 132:
        raise SystemExit("Base checkpoint is not the 1024-d MiniRGB + 132-d log-mel live model")

    audio_stats = ROOT / base_cfg.get("audio_stats", "data/features/audio_stats.json")
    quality_stats = ROOT / base_cfg.get("quality_stats", "data/features/quality_stats.json")
    train_ds = ReviewedWindowDataset("train", train_intervals, features, audio_stats, quality_stats, args.window)
    val_ds = ReviewedWindowDataset("val", val_intervals, features, audio_stats, quality_stats, args.window)
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("No supervised windows. Review positive and/or normal intervals in train and val.")
    train_y = np.concatenate([target[known] for target, known in train_ds.labels.values() if known.any()])
    if train_y.size == 0 or np.unique(train_y).size < 2:
        raise SystemExit("UCA train needs reviewed positive and normal snippets; positive-only fine-tuning is refused")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    y_val, _, _ = collect_predictions(model, val_loader, device)
    if y_val.size == 0 or np.unique(y_val).size < 2:
        raise SystemExit("UCA val must contain reviewed positive and normal snippets before model selection")

    # Restrict fine-tuning to the snippet classifier: the data is small, while the cross-modal
    # representation and clip-level head remain fixed to the checkpoint that already runs live.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.snippet_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.snippet_head.parameters(), lr=args.lr)

    run_dir = ROOT / "runs" / f"{date.today().isoformat()}_{args.run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    history_path.write_text("epoch,train_loss,val_loss,val_ap,seconds\n", encoding="utf-8")
    config = {**base_cfg, "run_tag": args.run_tag, "base_checkpoint": str(base_path.relative_to(ROOT)),
              "review_table": str(review_path.relative_to(ROOT)), "feature_manifest": str(manifest_path.relative_to(ROOT)),
              "training_method": "masked snippet BCE on reviewed UCA temporal intervals; snippet_head only",
              "window_snippets": args.window, "epochs": args.epochs, "lr": args.lr,
              "train_split": "UCA train only", "model_selection_split": "UCA val only", "test_split": "UCA test untouched"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    best_ap, best_epoch, best_metrics = -1.0, -1, {}
    for epoch in range(1, args.epochs + 1):
        model.eval()  # frozen encoders stay deterministic; only the linear snippet head updates
        start_time = time.perf_counter()
        train_losses = []
        for batch in train_loader:
            visual, audio = batch["visual"].to(device), batch["audio"].to(device)
            mask, known = batch["mask"].to(device), batch["known"].to(device)
            reliability = batch["reliability"].to(device)
            target = batch["target"].to(device)
            logits = model(visual, audio, mask, reliability)["snippet_logits"]
            loss = F.binary_cross_entropy_with_logits(logits[known], target[known])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        y_true, y_score, val_loss = collect_predictions(model, val_loader, device)
        val_ap = float(average_precision(y_true, y_score))
        seconds = time.perf_counter() - start_time
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch},{np.mean(train_losses):.6f},{val_loss:.6f},{val_ap:.6f},{seconds:.1f}\n")
        print(f"[epoch {epoch:02d}/{args.epochs}] train_loss={np.mean(train_losses):.4f} "
              f"val_loss={val_loss:.4f} val_snippet_ap={val_ap:.4f} [{seconds:.1f}s]", flush=True)
        metrics = {"epoch": epoch, "val_loss": val_loss, "val_snippet_ap": val_ap,
                   "val_reviewed_snippets": int(y_true.size), "val_positive_snippets": int(y_true.sum())}
        if val_ap > best_ap:
            best_ap, best_epoch, best_metrics = val_ap, epoch, metrics
            new_ckpt = {**ckpt, "model": model.state_dict(), "config": config, "epoch": epoch,
                        "val": metrics, "training_origin": str(base_path.relative_to(ROOT))}
            torch.save(new_ckpt, run_dir / "ckpt_best.pt")

    last_ckpt = {**ckpt, "model": model.state_dict(), "config": config, "epoch": args.epochs,
                 "val": best_metrics, "training_origin": str(base_path.relative_to(ROOT))}
    torch.save(last_ckpt, run_dir / "ckpt_last.pt")
    summary = {"best_epoch": best_epoch, "best": best_metrics, "training_windows": len(train_ds),
               "validation_windows": len(val_ds), "train_videos": len(train_intervals),
               "validation_videos": len(val_intervals), "test_used": False}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.relative_to(ROOT)), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
