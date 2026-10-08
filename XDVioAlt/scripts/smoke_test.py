#!/usr/bin/env python
"""SafeWatch environment smoke test.

Verifies that every core dependency imports, that PyTorch can actually run a
forward/backward/step on CPU, and measures a representative model step so we
know whether local CPU training is viable.

Run:
    .venv/bin/python scripts/smoke_test.py
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import time

CORE_MODULES = [
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("streamlit", "streamlit"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("cv2", "opencv-python-headless"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("huggingface_hub", "huggingface_hub"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
    ("pytest", "pytest"),
]

# Representative of the SafeWatch fusion head: conv1d over snippet features
# (1152 = 1024 RGB + 128 audio) for a 200-snippet sequence.
BENCH_BATCH = 8
BENCH_TIMESTEPS = 200
BENCH_IN_DIM = 1152
BENCH_ITERS = 5


def check_modules() -> int:
    print("=" * 72)
    print("1. MODULE IMPORTS")
    print("=" * 72)
    failures = 0
    for mod_name, pkg_name in CORE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "?")
            print(f"  [ok]   {pkg_name:22s} {version}")
        except Exception as exc:  # noqa: BLE001 - report anything as a failure
            print(f"  [FAIL] {pkg_name:22s} {type(exc).__name__}: {exc}")
            failures += 1
    return failures


def check_external_tools() -> int:
    print()
    print("=" * 72)
    print("2. EXTERNAL TOOLS")
    print("=" * 72)
    failures = 0
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        if path is None:
            print(f"  [FAIL] {tool:8s} not found in PATH")
            failures += 1
            continue
        out = subprocess.run(
            [tool, "-version"], capture_output=True, text=True, check=False
        ).stdout.splitlines()
        print(f"  [ok]   {tool:8s} {out[0] if out else path}")
    return failures


def bench_torch_step() -> float:
    """Time one full training step of a representative fusion head."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv1d(BENCH_IN_DIM, 512, 1),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Conv1d(512, 128, 1),
        nn.ReLU(),
        nn.Conv1d(128, 1, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(BENCH_BATCH, BENCH_IN_DIM, BENCH_TIMESTEPS)

    # warm-up (thread pool / lazy init)
    y = model(x)
    loss = y.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        y = model(x)
        loss = y.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (time.perf_counter() - start) / BENCH_ITERS


def check_torch_math() -> int:
    print()
    print("=" * 72)
    print("3. PYTORCH FUNCTIONAL CHECK")
    print("=" * 72)
    try:
        import torch

        torch.manual_seed(0)
        a = torch.randn(256, 256, requires_grad=True)
        b = torch.randn(256, 256)
        c = (a @ b).sum()
        c.backward()
        assert a.grad is not None, "backward() produced no gradient"
        assert torch.isfinite(c).item(), "matmul produced non-finite value"
        print(f"  [ok]   matmul+backward   threads={torch.get_num_threads()}")
        print(f"  [ok]   device            cpu (no CUDA available: "
              f"{not torch.cuda.is_available()})")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] pytorch functional check: {type(exc).__name__}: {exc}")
        return 1

    try:
        import onnxruntime as ort

        print(f"  [ok]   onnxruntime EPs   {ort.get_available_providers()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] onnxruntime: {type(exc).__name__}: {exc}")
        return 1

    step = bench_torch_step()
    print(f"  [ok]   training step     {step * 1000:.1f} ms "
          f"(batch={BENCH_BATCH}, T={BENCH_TIMESTEPS}, d={BENCH_IN_DIM})")
    # XD-Violence train split ~3950 videos -> ~31 steps/epoch at batch 128.
    est_epoch = step * 4 * 31
    print(f"  [info] rough epoch cost  ~{est_epoch:.1f} s for 31 larger steps "
          f"(batch=128) -> {est_epoch * 50 / 3600:.2f} h for 50 epochs")
    return 0


def main() -> int:
    print(f"python  : {sys.version.split()[0]}  ({sys.executable})")
    failures = check_modules()
    failures += check_external_tools()
    failures += check_torch_math()
    print()
    print("=" * 72)
    if failures:
        print(f"RESULT: {failures} problem(s) found - fix before continuing")
    else:
        print("RESULT: environment OK - ready for Session 2 (data bootstrap)")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
