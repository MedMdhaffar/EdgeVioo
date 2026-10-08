#!/usr/bin/env python
"""P7 - measure the **whole** edge pipeline on real video: decode -> sample -> extractor -> head.

The report claims "the head is not the bottleneck, extraction is" (``docs/report/rapport.md`` §4.4).
That claim was only ever checked on the *audio* side and on the head alone. This script closes it by
timing every stage on actual videos from the corpus, on the CPU that is really available (no GPU):

    probe      frame count / fps without decoding
    decode     sequential walk keeping one frame per snippet (seek-free, see edge.video)
    extractor  MiniRGB on the sampled frames (PyTorch, and the ONNX/int8 graph)
    head       the trained fusion head on the resulting (T, 1024) features

``--realtime`` divides the total by the clip duration, which is the only number that says whether
the edge track is plausible. Nothing is extrapolated to a Jetson: the report states the device.

Usage:
    .venv/bin/python scripts/edge_budget.py --videos 3
    .venv/bin/python scripts/edge_budget.py --videos 3 \\
        --ckpt runs/2026-09-29_p3vggish30ep_cross/ckpt_best.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.edge.extractor import MiniRGB, frame_grid  # noqa: E402
from safewatch.edge.video import probe, sample_frames  # noqa: E402

EXPORT_DIR = Path("export/onnx")


def _percentiles(samples: list[float]) -> dict:
    if not samples:
        return {}
    ordered = sorted(samples)
    return {"n": len(ordered), "mean": round(statistics.mean(ordered), 4),
            "p50": round(ordered[len(ordered) // 2], 4),
            "min": round(ordered[0], 4), "max": round(ordered[-1], 4)}



def _onnx_parity(path: Path, dummy: torch.Tensor, reference: np.ndarray) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = session.run(None, {"frames": dummy.numpy()})[0]
    return float(np.abs(out - reference).max())


def _onnx_latency(path: Path, dummy: torch.Tensor, threads: int, rounds: int = 30) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    feed = {"frames": dummy.numpy()}
    for _ in range(3):
        session.run(None, feed)
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        session.run(None, feed)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {"latency_ms": _percentiles(samples)}


def export_extractor(model: MiniRGB, width: int, threads: int, int8: bool,
                     parity_tolerance: float) -> dict:
    """ONNX + int8 for the extractor, with the same rules as the head export (parity, not hope)."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    size = model.input_size
    dummy = torch.rand(1, 3, size, size)
    stem = f"minirgb_w{width}"
    fp32_path = EXPORT_DIR / f"{stem}_fp32.onnx"
    report: dict = {"width": width, "input_size": size, "params": model.num_params}

    with torch.no_grad():
        reference = model(dummy).numpy().copy()
    torch.onnx.export(model, (dummy,), str(fp32_path), input_names=["frames"],
                      output_names=["features"], dynamic_axes={"frames": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    report["fp32"] = {"file": fp32_path.name,
                      "size_mb": round(fp32_path.stat().st_size / 2**20, 3),
                      "max_abs_diff_vs_pytorch": _onnx_parity(fp32_path, dummy, reference)}
    report["fp32"].update(_onnx_latency(fp32_path, dummy, threads))

    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = EXPORT_DIR / f"{stem}_int8.onnx"
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        report["int8"] = {
            "file": int8_path.name,
            "size_mb": round(int8_path.stat().st_size / 2**20, 3),
            "max_abs_diff_vs_pytorch": _onnx_parity(int8_path, dummy, reference),
            "parity_tolerance": parity_tolerance}
        report["int8"]["accepted"] = report["int8"]["max_abs_diff_vs_pytorch"] <= parity_tolerance
        report["int8"].update(_onnx_latency(int8_path, dummy, threads))
        report["size_ratio_int8_over_fp32"] = round(
            report["int8"]["size_mb"] / report["fp32"]["size_mb"], 3)
    return report


def measure_videos(paths: list[str], model: MiniRGB, threads: int, ckpt: str | None) -> dict:
    """Time each stage on real clips; returns per-clip rows plus the stage percentiles."""
    head = None
    head_kind = None
    head_dims: dict = {}
    if ckpt:
        from safewatch.eval.runner import load_model

        head, head_cfg, head_kind = load_model(ckpt, torch.device("cpu"))
        head.eval()
        head_dims = {"audio_dim": int(head_cfg.get("audio_dim", 128)),
                     "n_reliability": int(head_cfg.get("n_reliability",
                                                       head_cfg.get("reliability_dim", 0)))}
        # Warm-up call: the first forward spins up the thread pool and cost 33x the steady state on
        # this machine (0.34 s vs 0.01 s), which would otherwise be charged to the first clip.
        warm = torch.randn(1, 100, 1024)
        warm_mask = torch.ones(1, 100, dtype=torch.bool)
        with torch.no_grad():
            if head_kind == "fusion":
                head(warm, torch.zeros(1, 100, head_dims["audio_dim"]), warm_mask,
                     torch.zeros(1, 100, head_dims["n_reliability"]))
            else:
                head(warm, warm_mask)

    stages: dict[str, list[float]] = {"probe": [], "decode": [], "extractor": [], "head": []}
    rows: list[dict] = []
    # ImageNet-style standardisation: the extractor is trained on normalised frames, so the budget
    # must be measured on the same preprocessing the deployed path would use.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for path in paths:
        start = time.perf_counter()
        info = probe(path)
        stages["probe"].append(time.perf_counter() - start)

        n_snippets = max(1, int(np.ceil(info["frames"] / 16.0)))
        indices = frame_grid(info["frames"], n_snippets)
        start = time.perf_counter()
        frames = sample_frames(path, indices, size=model.input_size)
        decode_s = time.perf_counter() - start
        stages["decode"].append(decode_s)

        batch = torch.from_numpy(frames).permute(0, 3, 1, 2)
        batch = (batch - mean) / std
        start = time.perf_counter()
        with torch.no_grad():
            features = model(batch[None])[0]                      # (T, 1024)
        extractor_s = time.perf_counter() - start
        stages["extractor"].append(extractor_s)

        head_s = 0.0
        if head is not None:
            steps = features.shape[0]
            mask = torch.ones(1, steps, dtype=torch.bool)
            # The fusion head also reads an audio stream and reliability channels. No edge *audio*
            # extractor exists yet (log-mel is measured separately at ~3 s/clip), so those two are
            # priced on zero-filled, correctly-shaped tensors: the head's cost does not depend on
            # the values, and this script makes no accuracy claim about them.
            extra = None
            if head_kind == "fusion":
                extra = (torch.zeros(1, steps, head_dims["audio_dim"]),
                         torch.zeros(1, steps, head_dims["n_reliability"]))
            start = time.perf_counter()
            with torch.no_grad():
                if extra is None:
                    head(features[None], mask)
                else:
                    head(features[None], extra[0], mask, extra[1])
            head_s = time.perf_counter() - start
            stages["head"].append(head_s)

        total = stages["probe"][-1] + decode_s + extractor_s + head_s
        duration = info["frames"] / max(info["fps"], 1e-9)
        rows.append({
            "video": Path(path).name, "frames": info["frames"], "snippets": n_snippets,
            "duration_s": round(duration, 1), "decode_s": round(decode_s, 3),
            "extractor_s": round(extractor_s, 3), "head_s": round(head_s, 4),
            "total_s": round(total, 3),
            "realtime_factor": round(duration / max(total, 1e-9), 1)})
    return {"clips": rows, "stages": {k: _percentiles(v) for k, v in stages.items() if v}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=int, default=3, help="how many clips to measure")
    parser.add_argument("--glob", default="data/raw/xd-violence/data/video/test_videos/*.mp4")
    parser.add_argument("--width", type=int, default=24, help="MiniRGB width (16/24/32 measured)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ckpt", default=None, help="fusion checkpoint to price the head stage")
    parser.add_argument("--int8", action="store_true", help="also quantise the extractor to int8")
    parser.add_argument("--parity-tolerance", type=float, default=0.05)
    parser.add_argument("--extractor-weights", default=None,
                        help="distilled MiniRGB .pt to export; WITHOUT it the ONNX is exported "
                             "from a RANDOM init (the bug behind the meaningless live extractor)")
    parser.add_argument("--out", default="export/onnx/edge_budget_report.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    paths = sorted(glob.glob(args.glob))[: args.videos]
    if not paths:
        raise SystemExit(f"[edge] no videos matching {args.glob}")

    model = MiniRGB(width=args.width).eval()
    if args.extractor_weights:
        ckpt = torch.load(args.extractor_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[edge] extractor weights <- {args.extractor_weights}")
    print(f"[edge] device=cpu/{args.threads} threads | MiniRGB width={args.width} "
          f"params={model.num_params:,} | {len(paths)} clips")

    with torch.no_grad():
        budget = measure_videos(paths, model, args.threads, args.ckpt)
    extractor = export_extractor(model, args.width, args.threads, args.int8,
                                 args.parity_tolerance)

    report = {
        "device": f"cpu/{args.threads} threads",
        "kind": "edge_pipeline_budget",
        "video_glob": args.glob,
        "extractor": extractor,
        "stages": budget["stages"],
        "clips": budget["clips"],
        "note": ("decode is seek-free (one sequential walk); the extractor/head are priced on the "
                 "frames the deployed path would actually receive"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = (f"{'video':44s} {'dur':>6s} {'decode':>7s} {'extract':>8s} {'head':>7s} "
              f"{'total':>7s} {'xRT':>6s}")
    print(header)
    print("-" * len(header))
    for row in budget["clips"]:
        print(f"{row['video'][:44]:44s} {row['duration_s']:6.1f} {row['decode_s']:7.3f} "
              f"{row['extractor_s']:8.3f} {row['head_s']:7.4f} {row['total_s']:7.3f} "
              f"{row['realtime_factor']:6.1f}")
    stages = budget["stages"]
    head_p50 = stages["head"]["p50"] if "head" in stages else float("nan")
    print(f"[edge] stage p50: decode {stages['decode']['p50']:.3f}s | "
          f"extractor {stages['extractor']['p50']:.3f}s | head {head_p50:.4f}s")
    print(f"[edge] extractor fp32 {extractor['fp32']['size_mb']} MB, "
          f"p50 {extractor['fp32']['latency_ms']['p50']:.2f} ms/frame")
    print(f"[edge] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

