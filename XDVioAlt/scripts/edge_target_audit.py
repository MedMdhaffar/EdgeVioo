#!/usr/bin/env python
"""Record a reproducible compatibility audit on the real Edge target.

Run this *on the board*, after copying the two ONNX artefacts and the matching config::

    python3 scripts/edge_target_audit.py \
      --head export/onnx/2026-10-03_p3live_minirgbmel_fp32.onnx \
      --config runs/2026-10-03_p3live_minirgbmel/config.json

It intentionally does not invent a Jetson performance number.  It records the board/software
identity, verifies that ONNX Runtime can instantiate and execute the exact artefacts, and writes a
JSON report that belongs next to the target benchmark and tegrastats log.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip()
    return text or None


def _package(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _memory_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def _session_probe(path: Path, feeds: dict[str, np.ndarray]) -> dict:
    import onnxruntime as ort

    started = time.perf_counter()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - started) * 1e3
    started = time.perf_counter()
    outputs = session.run(None, feeds)
    infer_ms = (time.perf_counter() - started) * 1e3
    return {
        "inputs": [{"name": item.name, "shape": item.shape, "type": item.type}
                   for item in session.get_inputs()],
        "outputs": [{"name": item.name, "shape": item.shape, "type": item.type}
                    for item in session.get_outputs()],
        "load_ms": round(load_ms, 3),
        "dummy_infer_ms": round(infer_ms, 3),
        "outputs_finite": bool(all(np.isfinite(value).all() for value in outputs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", type=Path, default=Path("export/onnx/minirgb_w24_fp32.onnx"))
    parser.add_argument("--head", type=Path,
                        default=Path("export/onnx/2026-10-03_p3live_minirgbmel_fp32.onnx"))
    parser.add_argument("--config", type=Path,
                        default=Path("runs/2026-10-03_p3live_minirgbmel/config.json"))
    parser.add_argument("--out", type=Path, default=Path("data/logs/edge_target_audit.json"))
    args = parser.parse_args()
    missing = [str(path) for path in (args.extractor, args.head, args.config) if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required target artefacts: {', '.join(missing)}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    audio_dim = int(config.get("audio_dim", 132))
    reliability_dim = int(config.get("n_reliability", 0))
    import onnxruntime as ort

    head_feeds = {"visual": np.zeros((1, 100, 1024), dtype=np.float32),
                  "audio": np.zeros((1, 100, audio_dim), dtype=np.float32),
                  "mask": np.ones((1, 100), dtype=bool)}
    # Supply a reliability tensor only if this exact export declares it.
    head_session = ort.InferenceSession(str(args.head), providers=["CPUExecutionProvider"])
    if any(item.name == "reliability" for item in head_session.get_inputs()):
        head_feeds["reliability"] = np.zeros((1, 100, reliability_dim), dtype=np.float32)

    report = {
        "purpose": "target compatibility probe; not an end-to-end throughput benchmark",
        "platform": {"python": sys.version, "system": platform.platform(),
                     "machine": platform.machine(), "memory_mib": _memory_mib(),
                     "nv_tegra_release": _command(["cat", "/etc/nv_tegra_release"]),
                     "l4t_packages": _command(["dpkg-query", "-W", "nvidia-l4t-core",
                                                  "nvidia-jetpack", "nvidia-tensorrt"])},
        "runtime": {"onnxruntime": _package("onnxruntime"), "numpy": _package("numpy"),
                    "providers": ort.get_available_providers()},
        "artefacts": {"extractor": str(args.extractor), "head": str(args.head),
                      "config": str(args.config), "feature_contract":
                      "MiniRGB 1024-d visual + log-mel 132-d audio"},
        "extractor": _session_probe(args.extractor,
                                     {"frames": np.zeros((1, 3, 112, 112), dtype=np.float32)}),
        "head": _session_probe(args.head, head_feeds),
        "next_step": "Run edge_emulated_bench.py or a full capture under tegrastats; retain "
                     "and validate long-window aggregation before deployment sign-off.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[edge-target-audit] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
