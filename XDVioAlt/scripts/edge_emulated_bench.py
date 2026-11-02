#!/usr/bin/env python
"""Simulated Pi/Jetson-class edge budget: the full deployed chain, on constrained cores.

The subject accepts a deployment **"ou simulé" on a plateforme à ressources limitées**. This script
is that simulation, with the rules stated up front so nobody can read more into it than it claims:

* it times the **exact deployed artefacts** - the MiniRGB ONNX extractor, the selected fusion-head
  ONNX, and the full audio/explanation chain (ffmpeg decode, log-mel, quality channels, frozen
  YAMNet AED, motion saliency) - not a PyTorch prototype;
* it runs on this machine's cores, **pinned** (``taskset``) to emulate the core budget of a Pi 4 /
  Jetson Nano (single best core for the serial stages);
* the single-core numbers are then scaled to the board's core frequency (stated ratio, no other
  modelling) and labelled *estimates*. A Pi 4 with 4 cores running the 4 stages in parallel would
  do better than these estimates; the estimate is deliberately the pessimistic bound.

Nothing here claims the numbers are measured on a Pi or a Jetson.  In particular, a head whose
attention graph was traced at 100 snippets is padded only for a *cost* measurement.  Splitting a
long clip into those windows is not semantically equivalent to full-sequence attention, therefore
this script never reports a detection-accuracy or deployment-readiness conclusion for windowing.

Usage:
    .venv/bin/python scripts/edge_emulated_bench.py --clips 4            # pinned to 1 core
    .venv/bin/python scripts/edge_emulated_bench.py --clips 4 --all-cores
    .venv/bin/python scripts/edge_emulated_bench.py --clips 4 --cores 0-3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.edge import aed  # noqa: E402
from safewatch.edge.audio import decode_audio, mel_patch_stats, quality_frames  # noqa: E402
from safewatch.edge.video import probe, sample_frames  # noqa: E402
from safewatch.explain import media as media_mod  # noqa: E402

EXTRACTOR_ONNX = Path("export/onnx/minirgb_w24_fp32.onnx")
HEAD_ONNX = Path("export/onnx/2026-10-03_p3live_minirgbmel_fp32.onnx")
HEAD_CONFIG = Path("runs/2026-10-03_p3live_minirgbmel/config.json")
SALIENCE_SECONDS = 5            # the console renders at most 5 evidence snippets
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: core-frequency ratio this machine -> Pi 4 / Jetson Nano single core (Cortex-A72/A57 @ 1.5 GHz vs
#: the measured core). Stated in the output; change it if the host changes.
BOARD_FREQUENCY_RATIO = 0.375


def active_cores() -> str:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def relaunch_with_affinity(cores: str) -> None:
    """Re-exec the same command pinned to ``cores`` so the measurement is constrained."""
    cmd = ["taskset", "-c", cores, sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execvp(cmd[0], cmd)


def _percentiles(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {"n": len(ordered), "mean": round(statistics.mean(ordered), 4),
            "p50": round(ordered[len(ordered) // 2], 4),
            "min": round(ordered[0], 4), "max": round(ordered[-1], 4)}


def pick_clips(n: int, splits=("test",)) -> list[Path]:
    """Prefer clips of very different lengths so the budget is not a single-length artefact."""
    video_root = Path("data/raw/xd-violence/data/video")
    candidates: list[Path] = []
    for split in splits:
        with Path(f"data/lists_official/{split}.csv").open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                clip_id = row["clip_id"]
                for pattern in (f"test_videos/{clip_id}.mp4", f"{clip_id}.mp4"):
                    path = video_root / pattern
                    if path.exists():
                        candidates.append(path)
                        break
    if not candidates:
        candidates = sorted(video_root.rglob("*.mp4"))
    # probing every clip would take minutes: a 200-clip sample is enough for the length spread
    candidates = candidates[:200]
    durations = []
    for path in candidates:
        try:
            durations.append((probe(str(path))["frames"], path))
        except Exception:
            continue
    durations.sort(key=lambda item: item[0])
    # take one from each third of the length distribution
    chosen: list[Path] = []
    third = max(1, len(durations) // 3)
    for i in range(0, third * 3, third):
        if durations[i][1] not in chosen:
            chosen.append(durations[i][1])
    while len(chosen) < n:
        for _, path in durations:
            if path not in chosen:
                chosen.append(path)
                break
        else:
            break
    return chosen[:n]


def _read_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[bench] cannot read checkpoint config {path}: {exc}") from exc


def _live_audio(path: Path, n_snippets: int, stride: float, fps: float, cfg: dict):
    """Build the same live audio/reliability tensors as ``ui.state.analyze_video``.

    Keeping this small reference here avoids a deceptively cheap benchmark that feeds zeros to a
    multimodal head.  It deliberately retains the current whole-clip feature representation: that
    limitation is reported below rather than hidden by the timing harness.
    """
    from safewatch.data import reliability as R
    from safewatch.data.multimodal import load_audio_stats, patches_per_snippet, pool_audio

    waveform = decode_audio(path)
    raw_mel = mel_patch_stats(waveform)
    quality = quality_frames(waveform)
    audio_dim = int(cfg.get("audio_dim", 132))
    if raw_mel.shape[0]:
        audio = pool_audio(raw_mel, n_snippets, patches_per_snippet(stride, fps))
        stats_file = cfg.get("audio_stats")
        stats = load_audio_stats(stats_file) if stats_file else None
        if stats is not None and audio.shape[1] == stats[0].shape[0]:
            audio = (audio - stats[0]) / np.maximum(stats[1], 1e-6)
        if cfg.get("audio_norm") == "clip" or stats is None:
            audio = (audio - audio.mean(axis=0, keepdims=True)) / np.maximum(
                audio.std(axis=0, keepdims=True), 1e-4)
    else:
        audio = np.zeros((n_snippets, audio_dim), dtype=np.float32)
    if audio.shape[1] != audio_dim:
        audio = (audio[:, :audio_dim] if audio.shape[1] > audio_dim else
                 np.pad(audio, ((0, 0), (0, audio_dim - audio.shape[1]))))
    return waveform, np.asarray(audio, dtype=np.float32), quality, R


def measure_clip(path: Path, extractor, head, head_inputs: list[str], cfg: dict) -> dict:
    """Time every stage of the deployed chain on one clip."""
    stages: dict[str, float] = {}

    start = time.perf_counter()
    info = probe(str(path))
    stages["probe"] = time.perf_counter() - start

    fps = float(info.get("fps") or 24.0)
    n_snippets = max(1, int(np.ceil(info["frames"] / 16.0)))
    from safewatch.edge.extractor import frame_grid
    indices = frame_grid(info["frames"], n_snippets)

    start = time.perf_counter()
    frames = sample_frames(path, indices, size=112)
    stages["decode"] = time.perf_counter() - start

    start = time.perf_counter()
    # ``sample_frames`` already returns float RGB in [0, 1].  Dividing it by 255 here used to feed
    # an almost-black representation that was unlike the live extractor's input.
    normalised = ((frames - IMAGE_MEAN.reshape(1, 1, 1, 3))
                  / IMAGE_STD.reshape(1, 1, 1, 3))
    visual = extractor.run(None, {"frames": np.ascontiguousarray(
        normalised.transpose(0, 3, 1, 2).astype(np.float32))})[0]
    stages["extractor"] = time.perf_counter() - start

    # audio chain: the deployed edge path, including the AED and the saliency the card now carries
    start = time.perf_counter()
    waveform, audio, quality, reliability_mod = _live_audio(
        path, n_snippets, 16.0, fps, cfg)
    stages["audio_features"] = time.perf_counter() - start

    aed_s = 0.0
    if aed.yamnet_available() and waveform.size > 0:
        start = time.perf_counter()
        frames_aed = aed.yamnet_frames(waveform)
        tags = aed.snippet_tags(frames_aed, n_snippets, snippet_s=16.0 / fps)
        aed_s = time.perf_counter() - start
        del frames_aed, tags
    start = time.perf_counter()
    evidence_seconds = [round(i * 16.0 / fps + 8.0 / fps, 2)
                        for i in range(SALIENCE_SECONDS)
                        if i * 16.0 / fps + 8.0 / fps < info["frames"] / fps]
    motion = media_mod.motion_saliency(path, evidence_seconds, Path("data/logs"),
                                       prefix=f"{path.stem}_bench")
    stages["saliency"] = time.perf_counter() - start
    del motion

    # The historical export has its time axis baked at 100.  Long-window calls below are a runtime
    # *cost* probe only, not an aggregation algorithm: cross-attention over a chunk differs from
    # cross-attention over the clip.  The report makes that non-equivalence machine-readable.
    head_s = 0.0
    visual_c = visual.astype(np.float32)
    chunk = 100
    for offset in range(0, n_snippets, chunk):
        steps = chunk
        start = time.perf_counter()
        feed_visual = visual_c[offset:offset + chunk]
        if feed_visual.shape[0] < steps:
            pad = np.zeros((steps - feed_visual.shape[0], feed_visual.shape[1]), dtype=np.float32)
            feed_visual = np.concatenate([feed_visual, pad], axis=0)
        valid = min(chunk, n_snippets - offset)
        feed_audio = audio[offset:offset + chunk]
        if feed_audio.shape[0] < steps:
            feed_audio = np.pad(feed_audio, ((0, steps - feed_audio.shape[0]), (0, 0)))
        feeds = {"visual": feed_visual[None], "audio": feed_audio[None],
                 "mask": np.arange(steps)[None] < valid}
        if "reliability" in head_inputs:
            # This checkpoint uses a seven-channel gate.  Build the same quality-derived channels
            # as live inference; the visual feature stream is the ONNX output just measured.
            rel = reliability_mod.reliability_channels(
                quality, visual_c, n_snippets,
                reliability_mod.load_quality_stats(cfg.get("quality_stats")), 16.0 / fps / 0.96)
            rel = rel[offset:offset + chunk]
            if rel.shape[0] < steps:
                rel = np.pad(rel, ((0, steps - rel.shape[0]), (0, 0)))
            feeds["reliability"] = rel[None].astype(np.float32)
        head.run(None, feeds)
        head_s += time.perf_counter() - start
    stages["head"] = head_s
    del visual_c

    total = sum(stages.values()) + aed_s
    stages["aed"] = aed_s
    duration = info["frames"] / fps
    return {
        "video": path.name, "frames": info["frames"], "snippets": n_snippets,
        "duration_s": round(duration, 1),
        **{f"{name}_s": round(value, 4) for name, value in stages.items()},
        "total_s": round(total, 4),
        "realtime_factor": round(duration / max(total, 1e-9), 1),
        "head_windowing": "padded_100_snippet_cost_only" if n_snippets > 100 else "single_window",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--cores", default="0",
                        help="taskset core list for the pinned run (default: core 0)")
    parser.add_argument("--all-cores", action="store_true",
                        help="skip the pinning (host baseline)")
    parser.add_argument("--out", default=None, help="output JSON (default: data/logs/...)")
    parser.add_argument("--head", type=Path, default=HEAD_ONNX,
                        help="ONNX head trained on MiniRGB + log-mel features")
    parser.add_argument("--config", type=Path, default=HEAD_CONFIG,
                        help="config.json belonging to --head's checkpoint")
    args = parser.parse_args()

    if args.all_cores:
        pin = "all"
    else:
        pin = args.cores
        if active_cores() != pin:
            relaunch_with_affinity(pin)
            raise SystemExit(0)

    import onnxruntime as ort

    if not args.head.exists() or not EXTRACTOR_ONNX.exists():
        raise SystemExit(
            "[bench] selected ONNX artefact is missing; export the live checkpoint first")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    extractor = ort.InferenceSession(str(EXTRACTOR_ONNX), sess_options=options,
                                     providers=["CPUExecutionProvider"])
    cfg = _read_config(args.config)
    head = ort.InferenceSession(str(args.head), sess_options=options,
                                providers=["CPUExecutionProvider"])
    head_inputs = [item.name for item in head.get_inputs()]

    # Warm-up prices thread-pool start-up separately from the model (head chunk is fixed at 100).
    dummy_frames = np.zeros((4, 3, 112, 112), dtype=np.float32)
    extractor.run(None, {"frames": dummy_frames})
    warm_feeds = {"visual": np.zeros((1, 100, 1024), dtype=np.float32),
                  "audio": np.zeros((1, 100, int(cfg.get("audio_dim", 132))), dtype=np.float32),
                  "mask": np.ones((1, 100), dtype=bool)}
    if "reliability" in head_inputs:
        warm_feeds["reliability"] = np.zeros(
            (1, 100, int(cfg.get("n_reliability", 0))), dtype=np.float32)
    head.run(None, warm_feeds)

    clips = pick_clips(args.clips)
    print(f"[bench] {len(clips)} clips, cores={active_cores()} "
          f"({'pinned' if pin != 'all' else 'all cores'})")
    rows = [measure_clip(path, extractor, head, head_inputs, cfg) for path in clips]
    for row in rows:
        print(f"  {row['video'][:48]:48s} {row['duration_s']:6.1f}s video "
              f"-> {row['total_s']:7.3f}s total  (x{row['realtime_factor']})")

    stage_names = ("probe", "decode", "extractor", "audio_features", "aed", "saliency", "head")
    stages = {name: _percentiles([row[f"{name}_s"] for row in rows])
              for name in stage_names}
    totals = [row["total_s"] for row in rows]
    mean_total = sum(value["mean"] for value in stages.values())
    mean_duration = float(np.mean([row["duration_s"] for row in rows]))
    report = {
        "device": "host CPU, pinned if requested - NOT a Pi/Jetson measurement",
        "head": str(args.head),
        "config": str(args.config),
        "feature_contract": "MiniRGB 1024-d visual + live log-mel 132-d audio",
        "long_video_scoring": "UNVALIDATED: 100-snippet padded windows are timed only",
        "cores": active_cores(),
        "pin": pin,
        "clips": rows,
        "stages": stages,
        "total": _percentiles(totals),
        "realtime_factor_mean": round(float(np.mean([row["realtime_factor"] for row in rows])), 1),
        "board_estimate": {
            "note": (f"single-core x{BOARD_FREQUENCY_RATIO} "
                     "(1.5 GHz Cortex-A72/A57 vs host core) - pessimistic bound: "
                     "4 cores running the stages in parallel would be faster"),
            "est_total_s_mean": round(mean_total / BOARD_FREQUENCY_RATIO, 2),
            "est_realtime_factor": round(mean_duration / (mean_total / BOARD_FREQUENCY_RATIO), 1),
        },
    }
    out = Path(args.out or f"data/logs/edge_emulated_bench_{'all' if pin == 'all' else pin}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[bench] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
