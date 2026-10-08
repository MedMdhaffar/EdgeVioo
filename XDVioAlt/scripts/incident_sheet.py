#!/usr/bin/env python
"""P5 - incident sheet: score one clip, explain the alert, render its evidence.

This is the bridge between the research pipeline and the operator console (P6). It writes, per clip:

    <out>/<clip_id>_incident.json   the alert card (category, segments, evidence, contributions,
                                    confidence, caveats, concise text) - exactly what the UI renders
    <out>/<clip_id>_spectrogram.png log-mel spectrogram of the strongest segment (--media)
    <out>/frames/<clip_id>_*.jpg    the frames of the top-attention snippets (--media)

Everything is derived from the checkpoint and the clip; nothing is recomputed from the test ground
truth, so the same command works on a clip that has no annotation (the UI's "import a video" case
uses exactly this path).

Named sound events: when ``scripts/extract_aed_tags.py`` has written the YAMNet tags for the clip
(``data/features/aed/tags/<clip>.npz``), the card names the acoustic events of each segment
(frozen out-of-domain tagger - presented as corroboration, see ``docs/avancement.md`` §5).

Usage:
    .venv/bin/python scripts/incident_sheet.py --ckpt runs/2026-09-29_p3full_late/ckpt_best.pt \\
        --clip 78MHz__#00-04-30_00-05-45_label_B6-0-0 --media
    .venv/bin/python scripts/incident_sheet.py --ckpt ... --worst 5      # 5 highest-scoring clips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.data import reliability as R  # noqa: E402
from safewatch.data.dataset import CATEGORIES, load_rows  # noqa: E402
from safewatch.data.multimodal import MultimodalClipDataset  # noqa: E402
from safewatch.edge import aed  # noqa: E402
from safewatch.eval import calibration as C  # noqa: E402
from safewatch.eval.runner import checkpoint_config, load_model, resolve_stride  # noqa: E402
from safewatch.explain import media as media_mod  # noqa: E402
from safewatch.explain import report as report_mod  # noqa: E402


def _dataset(cfg: dict, test_csv: Path):
    """Same dataset construction the scorer uses, so a sheet describes the scored pipeline."""
    return MultimodalClipDataset(test_csv, cfg.get("audio_dir", "data/features/mel"),
                                 modality="both", audio_stats=cfg.get("audio_stats"),
                                 quality_dir=cfg.get("quality_dir"),
                                 quality_stats=cfg.get("quality_stats"),
                                 audio_norm=cfg.get("audio_norm", "none"),
                                 max_snippets=None, train=False, crop=cfg.get("crop", "mean"))


def _forward(model, item, kind: str, drop: str = "none"):
    visual = item["visual"][None].to("cpu")
    audio = item["audio"][None].to("cpu")
    mask = item["mask"][None].to("cpu")
    reliability = item.get("reliability")
    reliability = reliability[None].to("cpu") if reliability is not None else None
    if kind == "fusion":
        visual, audio, reliability = R.drop_modality(visual, audio, reliability, drop)
        return model(visual, audio, mask, reliability)
    features = item["visual"][None].to("cpu")
    return model(features, mask)


def build_context(model, cfg, kind, item, row, clip_id, stride, test_csv, quality_dir):
    """Run the model on one clip and collect every piece of evidence the card needs."""
    with torch.no_grad():
        out = _forward(model, item, kind)
        meta: dict = {"provenance": {"checkpoint": cfg.get("_ckpt"), "clip_id": clip_id,
                                     "stride_frames": stride,
                                     "fusion_mode": cfg.get("fusion_mode", "modality")}}
        for drop in ("audio", "visual"):
            if kind != "fusion":
                continue
            out_drop = _forward(model, item, kind, drop=drop)
            intact = torch.sigmoid(out["snippet_logits"])[0].numpy()
            without = torch.sigmoid(out_drop["snippet_logits"])[0].numpy()
            meta.setdefault("drop_one", {})[drop] = {
                "mean_probability_intact": round(float(intact.mean()), 4),
                "mean_probability_without": round(float(without.mean()), 4),
                "delta": round(float(intact.mean() - without.mean()), 4),
            }

    snippet_probability = torch.sigmoid(out["snippet_logits"])[0].numpy()
    attention = out.get("attn")
    alpha = out.get("alpha")
    category_probability = {
        name: float(torch.sigmoid(out["multi_logits"])[0][index])
        for index, name in enumerate(CATEGORIES)
    } if "multi_logits" in out else {}

    quality = None
    if quality_dir is not None:
        path = Path(quality_dir) / f"{clip_id}.npy"
        if path.exists():
            quality = np.asarray(np.load(path), dtype=np.float32)
    n_snippets = int(row["T"])
    reliability = (R.reliability_channels(quality, None, n_snippets)
                   if quality is not None else None)
    if quality is not None:
        from safewatch.data.multimodal import pool_audio
        reliability = pool_audio(quality, n_snippets)

    # Named sound events: YAMNet tag matrix computed offline by scripts/extract_aed_tags.py
    # (official-grid snippets). Absent -> the card falls back to signal statistics only.
    aed_tags = None
    aed_npz = aed.TAGS_DIR / f"{clip_id}.npz"
    if aed_npz.exists():
        aed_tags = np.asarray(np.load(aed_npz)["tags"], dtype=np.float32)

    return report_mod.ClipContext(
        clip_id=clip_id, stride_frames=float(stride), fps=float(cfg.get("fps", 24.0)),
        n_snippets=n_snippets, snippet_probability=snippet_probability,
        attention=attention.numpy()[0] if attention is not None else None,
        alpha=alpha.numpy()[0] if alpha is not None else None,
        # the per-modality heads of the late-fusion branch emit *logits*: sigmoid them here, or the
        # card quotes 7.28 as if it were a probability (caught by reading a real sheet, Session 10)
        snippet_visual=(torch.sigmoid(out["snippet_visual"])[0].numpy()
                        if out.get("snippet_visual") is not None else None),
        snippet_audio=(torch.sigmoid(out["snippet_audio"])[0].numpy()
                       if out.get("snippet_audio") is not None else None),
        category_probability=category_probability, reliability=reliability,
        video_path=str(media_mod.find_video(clip_id) or ""), meta=meta, aed_tags=aed_tags)



_shortlist: list = []


def _draw_motion_overlays(video: Path, seconds: list, frames: list, prefix: str,
                          frames_dir: Path) -> list[dict]:
    """Motion zones (subject: "les images **ou zones** significatives") on the key frames.

    The saliency is computed on the raw clip, never on the features: a box here is a *motion* zone,
    and the card says exactly that - the overlay JPEG is written next to the key frame, and the
    normalised boxes go into the card so the UI can draw them on any render size.
    """
    motion = media_mod.motion_saliency(video, seconds, frames_dir, prefix=prefix)
    by_second = {round(entry["second"], 2): entry for entry in motion}
    entries: list[dict] = []
    for index, frame in enumerate(frames):
        entry = by_second.get(frame["second"])
        if entry is None or not entry["boxes"]:
            continue
        import cv2

        image = cv2.imread(frame["path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        for box in entry["boxes"]:
            x0, y0 = int(box["x"] * width), int(box["y"] * height)
            x1, y1 = int((box["x"] + box["w"]) * width), int((box["y"] + box["h"]) * height)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 128), 2)
        overlay = frames_dir / f"{prefix}_zones_{index:02d}_{frame['second']:07.2f}s.jpg"
        cv2.imwrite(str(overlay), image)
        entry = dict(entry)
        entry["overlay"] = str(overlay)
        entries.append(entry)
    return entries


def _write(card: dict, context, config, out_dir: Path, with_media: bool) -> None:
    """Attach the artefacts and write the sheet (JSON always, PNG/JPG only with ``--media``)."""
    if with_media and context.video_path:
        video = Path(context.video_path)
        frames_dir = out_dir / "frames"
        snippets = [snippet["snippet"] for snippet in
                    card["visual_evidence"].get("top_snippets", [])][: config.top_k_evidence]
        seconds = media_mod.frame_times(snippets, context.stride_frames, context.fps)
        frames = media_mod.key_frames(video, seconds, frames_dir, prefix=context.clip_id)
        card["media"]["key_frames"] = frames
        if frames:
            zones = _draw_motion_overlays(video, seconds, frames, context.clip_id, frames_dir)
            if zones:
                card["media"]["motion"] = zones
        if card["segments"]:
            strongest = max(card["segments"], key=lambda s: s["peak_probability"])
            path = media_mod.spectrogram(
                video, out_dir / f"{context.clip_id}_spectrogram.png",
                start_s=strongest["start_s"], end_s=strongest["end_s"],
                title=f"{context.clip_id} [{strongest['start_s']:.1f}-{strongest['end_s']:.1f} s]")
            if path:
                card["media"]["spectrogram"] = str(path)
    elif with_media:
        card["media"]["note"] = "clip mp4 not found on disk; no frames or spectrogram rendered"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{context.clip_id}_incident.json"
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(report_mod.summarise(card))
    print(f"  -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--clip", default=None, help="clip id (default: use --worst)")
    parser.add_argument("--worst", type=int, default=0,
                        help="instead of --clip, the N highest-scoring clips of the test split")
    parser.add_argument("--out", default="sheets")
    parser.add_argument("--media", action="store_true",
                        help="also render key frames + spectrogram (needs the clip's mp4 on disk)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--max-gap-seconds", type=float, default=2.5)
    parser.add_argument("--top-k", type=int, default=5, help="evidence snippets to keep")
    parser.add_argument("--language", default="fr", choices=("fr", "en"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cfg = dict(checkpoint_config(args.ckpt))
    cfg["_ckpt"] = args.ckpt
    kind = "fusion" if cfg.get("fusion_mode") else "mil"
    stride = resolve_stride(cfg, None)
    test_csv = Path(cfg.get("lists_dir") or "data/lists") / "test.csv"
    rows = {row["clip_id"]: row for row in load_rows(test_csv)}
    model, full_cfg, _kind = load_model(args.ckpt, torch.device("cpu"))
    cfg.update({k: v for k, v in full_cfg.items() if k not in cfg})

    out_dir = Path(args.out)
    dataset = _dataset(cfg, test_csv)
    config = report_mod.AlertConfig(threshold=args.threshold, min_seconds=args.min_seconds,
                                   max_gap_seconds=args.max_gap_seconds, top_k_evidence=args.top_k,
                                   language=args.language)
    calibration = C.CalibrationMap.load(Path(args.ckpt).parent / "calibration.json")
    if calibration is None:
        print("[sheet] no calibration.json next to the checkpoint: confidences stay uncalibrated "
              "(run scripts/fit_calibration.py to fit one)")

    rendered = 0
    for item in dataset:
        clip_id = item["clip_id"]
        if args.clip and clip_id != args.clip:
            continue
        context = build_context(model, cfg, kind, item, rows[clip_id], clip_id, stride,
                                test_csv, cfg.get("quality_dir"))
        card = report_mod.build_alert(context, config, calibration)
        if args.worst and not args.clip:
            _shortlist.append((card["confidence"]["raw_clip_probability"], context, card))
            continue
        _write(card, context, config, out_dir, args.media)
        rendered += 1
    if args.worst and not args.clip:
        ranked = sorted(_shortlist, key=lambda row: row[0], reverse=True)
        for _score, context, card in ranked[: args.worst]:
            _write(card, context, config, out_dir, args.media)
            rendered += 1
    if rendered == 0:
        print("[sheet] nothing written - unknown clip id?")
        return 1
    print(f"[sheet] {rendered} incident sheet(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
