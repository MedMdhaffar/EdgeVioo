"""Score a trained checkpoint on the XD-Violence test split and report metrics.

Usage:
    .venv/bin/python -m safewatch.eval.runner --ckpt runs/2026-09-23_p1_visual/ckpt_best.pt

Outputs (written next to the checkpoint by default):
    predictions.npz  - per-clip snippet score curves + snippet ground truth
    test_metrics.json - both AP protocols x both metrics, localisation report, per-class AP

Important: frame annotations are converted to snippet indices with the *measured* grid rule
(``--stride-frames``, default 64, i.e. ``T = ceil(nb_frames / 64)`` as verified by
``scripts/check_feature_grid.py``). Getting this rule wrong silently destroys AP.

"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from safewatch.data import reliability as R
from safewatch.data.dataset import CATEGORIES, N_CATEGORIES, ClipDataset, load_rows
from safewatch.data.multimodal import (
    AUDIO_DIM,
    DEFAULT_QUALITY_DIR,
    DEFAULT_QUALITY_STATS,
    MultimodalClipDataset,
    audit_audio_rows,
    report_audio_preflight,
)
from safewatch.eval import metrics as M
from safewatch.models.fusion import FusionModel
from safewatch.models.mil import MILModel

IN_DIMS = {"swin_rgb": 768, "i3d_rgb": 2048, "c3d_rgb": 4096, "i3d_official": 1024}

# Frames per snippet, per grid: measured on the HF mirror (scripts/check_feature_grid.py) and per
# the official release (16-frame snippets). Unknown feature sets must pass --stride-frames
# explicitly: guessing the grid silently destroys AP (docs/journal.md, defect 1).
STRIDE_FRAMES = {"swin_rgb": 64.0, "i3d_official": 16.0}


def model_kind(cfg: dict) -> str:
    """``"fusion"`` for a safewatch.train_fusion checkpoint, ``"mil"`` for a unimodal one."""
    return "fusion" if cfg.get("fusion_mode") else "mil"


def checkpoint_config(ckpt_path: str | Path) -> dict:
    """Config stored inside a checkpoint (used to pick the test list / stride before scoring)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return dict(ckpt.get("config", {}))


def infer_in_dim(state: dict) -> int | None:
    """Recover the model input width from the weights, instead of trusting a name lookup table.

    Handles ``MILModel`` (``encoder.net.0.weight``), the non-early fusion variants
    (``visual_encoder`` / ``audio_encoder``) and early fusion (concatenated input width).
    """
    for name in ("encoder.net.0.weight", "visual_encoder.net.0.weight",
                 "audio_encoder.net.0.weight"):
        tensor = state.get(name)
        if tensor is not None and tensor.ndim >= 2:
            return int(tensor.shape[1])
    return None


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict, str]:
    """Rebuild a checkpoint -> ``(model, config, kind)``; ``kind`` is ``"mil"`` or ``"fusion"``."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}))
    state = ckpt["model"]
    kind = model_kind(cfg)
    common = {"emb": int(cfg.get("emb", 128)), "hidden": int(cfg.get("hidden", 512)),
              "dropout": float(cfg.get("dropout", 0.6))}

    if kind == "fusion":
        model: torch.nn.Module = FusionModel(
            visual_dim=int(cfg.get("visual_dim", 768)),
            audio_dim=int(cfg.get("audio_dim", AUDIO_DIM)), n_categories=N_CATEGORIES,
            fusion_mode=str(cfg["fusion_mode"]), heads=int(cfg.get("heads", 4)),
            n_reliability=int(cfg.get("n_reliability", 0)),
            gate_source=str(cfg.get("gate_source", "embeddings")), **common)
    else:
        in_dim = int(cfg.get("in_dim") or 0) or infer_in_dim(state) or (
            AUDIO_DIM if cfg.get("modality") == "audio"
            else IN_DIMS.get(cfg.get("feature_set", "swin_rgb"), 768))
        model = MILModel(in_dim=in_dim, **common)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg, kind


def load_intervals(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    """Read ``data/annotations/test_intervals.csv`` -> {clip_id: [(start, end), ...]}."""
    intervals: dict[str, list[tuple[int, int]]] = {}
    csv_path = Path(path)
    if not csv_path.exists():
        return intervals
    import csv

    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            intervals.setdefault(row["clip_id"], []).append((int(row["start"]), int(row["end"])))
    return intervals


@torch.no_grad()
def _causal_prefix_ends(n: int, stride: int) -> list[int]:
    """1-based prefix endpoints for strided causal scoring (always includes the full clip)."""
    ends = list(range(int(stride), n, int(stride))) + [n]
    return [e for e in ends if e >= 1]


def _prefix_tensors(tensors: tuple[torch.Tensor, ...], end: int) -> tuple[torch.Tensor, ...]:
    """Slice every ``(1, T, ...)`` tensor to its first ``end`` steps (batch dim untouched)."""
    return tuple(t[:, :end] for t in tensors if t is not None)


def _forward_tensors(model: torch.nn.Module, kind: str,
                     tensors: tuple[torch.Tensor, ...]) -> dict:
    """One forward pass from a stored tensor tuple (fusion takes 4, unimodal takes 1)."""
    if kind == "fusion":
        visual, audio, mask, reliability = tensors
        return model(visual, audio, mask, reliability)
    (features,) = tensors
    return model(features)


@torch.no_grad()
def score_clips(ckpt_path: str | Path, test_csv: str | Path, device: torch.device,
                drop: str = "none", perturb=None,
                causal_prefix_stride: int = 0, progress_every: int = 0,
                stride_frames: float | None = None,
                ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                           dict[str, np.ndarray]]:
    """Run the model on every test clip (full length) -> snippet scores, masks, multi-logits.

    ``causal_prefix_stride`` re-scores strided prefixes (``--causal`` CLI): only the score at
    each evaluated prefix end is kept and step-held forward, so snippet ``t`` never sees the
    future. The final evaluated prefix is always the full clip. Cost scales as ``T / stride``
    forwards per clip (~23x at stride 8 on the official grid, vs ~182x exact).

    ``drop`` removes one modality for the robustness study ("audio" or "visual"): its features are
    zeroed **and** its reliability channels are set to the degraded sentinel, exactly as modality
    dropout does during training (:mod:`safewatch.data.reliability`).

    ``perturb`` is the degradation hook used by ``scripts/robustness_sweep.py``: a callable
    ``(visual, audio, reliability) -> (visual, audio, reliability)`` applied *inside* this loop, so
    a degraded score comes from the same code path as the clean one (no second scorer to drift out
    of sync - the defect behind the wrong VGGish read-out, ``docs/journal.md`` Session 10).
    For a unimodal checkpoint the single stream arrives as ``visual`` with ``audio=None``; the hook
    must ignore ``None`` streams.
    """
    model, cfg, kind = load_model(ckpt_path, device)
    modality = "both" if kind == "fusion" else cfg.get("modality", "visual")
    crop = cfg.get("crop", "mean")
    if modality == "visual":
        dataset = ClipDataset(test_csv, max_snippets=None, train=False, crop=crop)
    else:
        dataset = MultimodalClipDataset(test_csv, cfg.get("audio_dir", "data/features/mel"),
                                        modality=modality, audio_stats=cfg.get("audio_stats"),
                                        quality_dir=cfg.get("quality_dir", DEFAULT_QUALITY_DIR),
                                        quality_stats=cfg.get("quality_stats",
                                                              DEFAULT_QUALITY_STATS),
                                        audio_norm=cfg.get("audio_norm", "none"),
                                        audio_grid=cfg.get("audio_grid", "patch"),
                                        max_snippets=None, train=False, crop=crop)
        audit_stride = (stride_frames or STRIDE_FRAMES.get(cfg.get("feature_set", "swin_rgb"),
                                                           64.0))
        expected_audio_dim = (int(cfg.get("audio_dim", AUDIO_DIM)) if kind == "fusion" else
                              int(cfg.get("in_dim") or infer_in_dim(model.state_dict()) or
                                  AUDIO_DIM))
        report_audio_preflight(
            audit_audio_rows(dataset.rows, cfg.get("audio_dir", "data/features/mel"),
                             cfg.get("audio_grid", "patch"), expected_audio_dim, audit_stride,
                             quality_dir=cfg.get("quality_dir", DEFAULT_QUALITY_DIR)),
            "evaluation")
    key = "audio" if modality == "audio" else "features"
    t_start = time.time()
    scores: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    multilogits: dict[str, np.ndarray] = {}
    for index, item in enumerate(dataset):
        if progress_every and index % progress_every == 0:
            print(f"[score] clip {index} t={time.time() - t_start:.0f}s",
                  flush=True)
        if kind == "fusion":
            visual = item["visual"][None].to(device)
            audio = item["audio"][None].to(device)
            mask = item["mask"][None].to(device)
            reliability = item.get("reliability")
            reliability = reliability[None].to(device) if reliability is not None else None
            visual, audio, reliability = R.drop_modality(visual, audio, reliability, drop)
            if perturb is not None:
                visual, audio, reliability = perturb(visual, audio, reliability)
            base = (visual, audio, mask, reliability)
        else:
            features = item[key][None].to(device)  # (1, T, D)
            if drop == modality:                   # modality lost: the model sees only zeros
                features = torch.zeros_like(features)
            elif perturb is not None:              # unimodal: single stream arrives as "visual"
                features, _audio, _rel = perturb(features, None, None)
            base = (features,)
        n = int(item["n_snippets"])
        if not causal_prefix_stride:
            out = _forward_tensors(model, kind, base)
            scores[item["clip_id"]] = torch.sigmoid(out["snippet_logits"])[0, :n].cpu().numpy()
        else:
            curve = np.zeros(n, dtype=np.float32)
            anchored = False
            for end in _causal_prefix_ends(n, causal_prefix_stride):
                tensors = base if end >= n else _prefix_tensors(base, end)
                out = _forward_tensors(model, kind, tensors)
                s = float(torch.sigmoid(out["snippet_logits"])[0, end - 1].cpu())
                curve[end - 1:] = s  # hold the streaming value until the next evaluated prefix
                anchored = anchored or end >= n
            assert anchored  # _causal_prefix_ends always includes the full clip
            scores[item["clip_id"]] = curve
        valid[item["clip_id"]] = np.ones(n, dtype=bool)
        multilogits[item["clip_id"]] = torch.sigmoid(out["multi_logits"])[0].cpu().numpy()
    return scores, valid, multilogits


def build_gt(rows: list[dict], intervals: dict[str, list[tuple[int, int]]],
             stride_frames: float) -> dict[str, np.ndarray]:
    """Per-clip snippet-level ground truth from frame intervals (measured grid rule)."""
    gt: dict[str, np.ndarray] = {}
    for row in rows:
        gt[row["clip_id"]] = M.snippet_labels(int(row["T"]), stride_frames,
                                              intervals.get(row["clip_id"], []))
    return gt


def compute_metrics(scores: dict[str, np.ndarray], gt: dict[str, np.ndarray],
                    gt_segments: dict[str, list[tuple[int, int]]],
                    multilogits: dict[str, np.ndarray], rows: list[dict],
                    stride_frames: float, threshold: float, min_length: int,
                    max_gap: int, fps: float = 24.0) -> dict:
    """Both AP protocols x both metrics, localisation report, clip-level per-class AP."""
    report: dict = {
        "n_clips": len(scores),
        "n_snippets": int(sum(v.size for v in scores.values())),
        "stride_frames": stride_frames,
        "segment_rule": {"threshold": threshold, "min_length": min_length, "max_gap": max_gap},
        "ap": {},
        "prevalence": M.prevalence(np.concatenate([gt[k] for k in sorted(gt)])),
    }
    for protocol in ("global", "per_video"):
        for metric in ("pr_auc", "average_precision"):
            result = M.evaluate(scores, gt, protocol=protocol, metric=metric)
            report["ap"][f"{protocol}__{metric}"] = result["ap"]

    pred_segments = {cid: M.segments_from_scores(s, threshold, min_length, max_gap)
                     for cid, s in scores.items()}
    report["localization"] = M.localization_report(pred_segments, gt_segments,
                                                  stride=stride_frames, fps=fps)

    per_class: dict[str, dict] = {}
    for index, category in enumerate(CATEGORIES):
        y = np.array([1.0 if category in (r["labels"] or "").split("|") else 0.0 for r in rows])
        s = np.array([float(multilogits[r["clip_id"]][index]) for r in rows])
        per_class[category] = {"ap": M.average_precision(y, s), "n_positive": int(y.sum())}
    report["per_class_clip_level_ap"] = per_class
    # a category with no positive clip in the split has AP = NaN: exclude it instead of letting
    # np.nanmean warn on an all-NaN slice (happens on subsets, e.g. a smoke test on 5 clips)
    valid = [v["ap"] for v in per_class.values() if not np.isnan(v["ap"])]
    report["macro_per_class_ap"] = float(np.mean(valid)) if valid else float("nan")
    return report


def default_test_csv(cfg: dict) -> Path:
    """Test list implied by the checkpoint: ``<lists_dir>/test.csv`` if present, else data/lists."""
    candidate = Path(cfg.get("lists_dir", "data/lists")) / "test.csv"
    return candidate if candidate.exists() else Path("data/lists/test.csv")


def resolve_stride(cfg: dict, explicit: float | None) -> float:
    """Frames per snippet: explicit flag > grid table (``feature_set``) > loud failure.

    Fusion checkpoints predating the ``feature_set`` field (the four 2026-09-24 P3 runs) fall
    back to the mirror grid, which is where their ``data/lists_partial`` visual paths come from.
    """
    if explicit:
        return float(explicit)
    feature_set = cfg.get("feature_set") or ("swin_rgb" if cfg.get("fusion_mode") else None)
    stride = STRIDE_FRAMES.get(str(feature_set))
    if stride is None:
        raise SystemExit(f"[score] no known grid for feature_set={cfg.get('feature_set')!r}; "
                         f"pass --stride-frames explicitly (known: {STRIDE_FRAMES})")
    return stride


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="checkpoint produced by safewatch.train")
    parser.add_argument("--test-csv", default=None,
                        help="default: <lists_dir of the checkpoint>/test.csv")
    parser.add_argument("--intervals", default="data/annotations/test_intervals.csv")
    parser.add_argument("--stride-frames", type=float, default=None,
                        help="frames per snippet; default: inferred from the checkpoint "
                             f"(known grids: {STRIDE_FRAMES})")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="video fps, used to express the detection delay in seconds")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-length", type=int, default=2,
                        help="minimum segment length in snippets (prefer --min-seconds)")
    parser.add_argument("--max-gap", type=int, default=1,
                        help="merge segments separated by fewer than this many snippets")
    parser.add_argument("--min-seconds", type=float, default=None,
                        help="minimum segment length in SECONDS (overrides --min-length); "
                             "grid-independent, so published numbers stay comparable")
    parser.add_argument("--max-gap-seconds", type=float, default=None,
                        help="merge gap in SECONDS (overrides --max-gap)")
    parser.add_argument("--drop-modality", default="none", choices=("none", "audio", "visual"),
                        help="robustness study: score with one modality removed (features zeroed "
                             "+ reliability flagged) -> predictions_drop_<modality>.npz")
    parser.add_argument("--causal-prefix-stride", type=int, default=0, metavar="K",
                         help="streaming score: re-forward strided prefixes (K=8 honest, exact is "
                              "K=1) and step-hold; snippet t never sees the future. "
                              "-> predictions_causal<K>.npz")
    parser.add_argument("--causal-progress", type=int, default=0, metavar="N",
                        help="print a progress line every N clips while scoring (causal runs are "
                             "long; 0 = silent)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=None, help="output dir (default: folder of --ckpt)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = checkpoint_config(args.ckpt)
    kind = model_kind(cfg)
    if args.drop_modality != "none" and kind != "fusion" and args.drop_modality != cfg.get(
            "modality", "visual"):
        raise SystemExit(f"[score] a {kind}/{cfg.get('modality', 'visual')} checkpoint cannot drop "
                         f"{args.drop_modality}: it has no such branch")
    test_csv = Path(args.test_csv) if args.test_csv else default_test_csv(cfg)
    stride = resolve_stride(cfg, args.stride_frames)
    min_length = (M.snippets_for_seconds(args.min_seconds, stride, args.fps)
                  if args.min_seconds is not None else args.min_length)
    max_gap = (M.snippets_for_seconds(args.max_gap_seconds, stride, args.fps, minimum=0)
               if args.max_gap_seconds is not None else args.max_gap)
    print(f"[score] ckpt={args.ckpt} kind={kind} modality={cfg.get('modality', 'both')} "
          f"test={test_csv} stride={stride:g} frames/snippet rule=({min_length},{max_gap}) snippets"
          f" = ({min_length * stride / args.fps:.2f},{max_gap * stride / args.fps:.2f}) s"
          f"{'' if args.drop_modality == 'none' else ' drop=' + args.drop_modality}"\
           f"{'' if not args.causal_prefix_stride else ' c=' + str(args.causal_prefix_stride)}")

    rows = load_rows(test_csv)
    intervals = load_intervals(args.intervals)
    scores, _valid, multilogits = score_clips(args.ckpt, test_csv, device,
                                              drop=args.drop_modality,
                                              causal_prefix_stride=args.causal_prefix_stride,
                                              progress_every=args.causal_progress,
                                              stride_frames=stride)
    gt = build_gt(rows, intervals, stride)
    gt_segments = {cid: M.gt_segments_from_intervals(intervals.get(cid, []), stride)
                   for cid in scores}

    report = compute_metrics(scores, gt, gt_segments, multilogits, rows, stride,
                             args.threshold, min_length, max_gap, fps=args.fps)
    report["segment_rule"].update({
        "min_seconds": (args.min_seconds if args.min_seconds is not None
                        else round(min_length * stride / args.fps, 3)),
        "max_gap_seconds": (args.max_gap_seconds if args.max_gap_seconds is not None
                            else round(max_gap * stride / args.fps, 3))})
    report["dropped_modality"] = args.drop_modality
    suffix = "" if args.drop_modality == "none" else f"_drop_{args.drop_modality}"
    causal_tag = ("" if not args.causal_prefix_stride else f"_causal{args.causal_prefix_stride}")
    suffix = f"{suffix}{causal_tag}"
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"predictions{suffix}.npz",
                        **{f"scores::{k}": v for k, v in scores.items()},
                        **{f"gt::{k}": v for k, v in gt.items()})
    (out_dir / f"test_metrics{suffix}.json").write_text(json.dumps(report, indent=2),
                                                       encoding="utf-8")

    summary = {k: v for k, v in report.items() if k != "per_class_clip_level_ap"}
    print(json.dumps(summary, indent=2))
    print("per-class clip-level AP:",
          {k: round(v["ap"], 4) for k, v in report["per_class_clip_level_ap"].items()})
    print(f"[ok] predictions + metrics -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
