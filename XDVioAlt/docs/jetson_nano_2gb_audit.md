# Jetson Nano 2GB readiness audit

**Audit date:** 2026-10-08  
**Target:** NVIDIA Jetson Nano 2GB Developer Kit  
**Physical board access:** unavailable in this workspace; the results below are a code and host audit,
not measurements from the Nano.

## Findings

### A compatible Edge checkpoint exists

The latest benchmark checkpoint (`2026-10-08_ucf_abuse_sourceval_baseline_s43`) was trained on
CLIP ViT-B-32 features and is not compatible with the live MiniRGB feature extractor, even though
both have a 1024-wide output. There is, however, an Edge-path checkpoint:
`runs/2026-10-03_p3live_minirgbmel/ckpt_best.pt`. Its config uses the MiniRGB feature list, 1024
visual values, 132 log-mel values, and clip-level audio normalization. This is the relevant
checkpoint for a Nano feasibility prototype; it is an older run and should not be confused with the
newer CLIP checkpoint's test metrics.

### Runtime package set does not match the Nano software stack

This repository's frozen environment is x86 CPU-only and currently pins Python 3.14, PyTorch
2.14 CPU, ONNX Runtime 1.30, NumPy 2.5, and OpenCV 5. The Nano 2GB is supported by the JetPack 4.x
line; NVIDIA's JetPack 4.6.5 release is based on Jetson Linux 32.7.5. The JetPack 4.6.1 component
list specifies CUDA 10.2 and TensorRT 8.2.1 for this generation of Jetson. The current lockfile
does not provide an ARM64/JetPack-specific set of these packages. Do not attempt to install the
project's lockfile as-is on the Nano; the deployment needs a separate, minimal environment using
versions available for the board's JetPack image. NVIDIA identifies the Nano 2GB kit as end-of-life
and recommends a swap file because the kit has only 2GB of physical memory.

References: [NVIDIA JetPack 4.6.5](https://developer.nvidia.com/jetpack-sdk-465),
[NVIDIA Jetson Nano 2GB getting started](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit).

### Whole-video preprocessing can exceed 2GB

The live path previously materialized every sampled video frame and then created a second normalized
float32 frame array. This was refactored: `iter_sample_frames` now decodes sequentially and live
inference runs MiniRGB on batches of 16 frames. At 112×112 RGB, each float32 frame is 150,528 bytes;
the two current batch buffers are about 4.6 MiB, independent of video duration. Extracted visual
features are retained because the current fusion model scores the sequence as a whole; that feature
array is much smaller than RGB inputs.

The audio and model paths are still whole-clip operations. Before the frame change, a two-hour video
at one sampled frame per 16 source frames at 24 fps (10,800 snippets) would need about 1.51 GiB per
full RGB float32 frame array (about 3.03 GiB for the sampled and normalized arrays), before other
memory. The frame arrays no longer scale this way, but whole-waveform/audio-feature processing can
still be large.

Audio is still decoded for the entire video into a float32 waveform. Two hours at 16 kHz is about
440 MiB for the waveform alone. Log-mel processing creates additional arrays. The model's
cross-attention also sees the whole snippet sequence. Further work must bound audio preprocessing
and establish a validated windowed scoring/aggregation method; simply splitting model calls changes
the global attention and clip score. Swap is not a real-time memory solution.

For scale, a 16-frame extraction batch uses about 4.6 MiB for sampled and normalized frames. The
features for a 100-snippet model window are small by comparison: approximately 0.39 MiB for 1024-d
visual features, 0.05 MiB for 132-d audio features, plus masks/reliability.

### Host timing is only a preliminary reference

The Edge-compatible MiniRGB + head path was measured on this AMD Ryzen 5 host, not on the Nano.
Three sample clips took 0.43–1.74 seconds end-to-end for 69–290 seconds of source video, but the
head timing used PyTorch and the audio stream was zero-filled. This is not a full audio-video
pipeline measurement and cannot be extrapolated as Nano throughput.

The Edge checkpoint's ONNX head export is 4.089 MB FP32 and measured 1.379 ms p50 per synthetic
100-snippet window on the host. FP16 is 2.079 MB and 1.458 ms p50 there. INT8 per-channel misses
the export logit tolerance (0.0662 > 0.05) and is rejected pending validation/calibration. The
MiniRGB FP32 ONNX artifact is 1.595 MB. All these figures are host CPU results only.

The exported cross-attention graph also has a practical fixed-window issue: attention reshapes were
traced at 100 snippets, and direct calls on shorter sequences failed. Windowed long-video scoring
and aggregation have not been validated for the Edge checkpoint.

## Readiness decision

**Not ready for a Nano 2GB deployment yet.** There is a compatible Edge-path model, but the package
lock is for a different platform, audio buffers and temporal model context still scale with clip
length, streaming/window aggregation is unverified, and there are no on-board memory or throughput
measurements. The FP32 model files are small; process memory and full media preprocessing are the
limiting risks.

## Next implementation and board checks

1. Stream audio decode/log-mel and establish validated windowed model scoring; the current model
   uses full-sequence cross-attention, so chunk aggregation requires validation and possibly retraining.
2. Export and run the Edge checkpoint in the actual Nano-compatible runtime. Check all required
   ONNX operators on the JetPack 4.x runtime; compare binary probabilities with the PyTorch
   reference on the same held-out validation clips.
3. On the Nano, record board details and run the full app under `tegrastats`; include model load,
   ffmpeg decode, log-mel, MiniRGB, fusion head, explanations, and UI overhead. Report peak RAM,
   p50/p95 window latency, sustained snippets/s, and end-to-end real-time factor.

Board facts needed to complete the hardware audit:

```bash
cat /etc/nv_tegra_release
uname -a
python3 --version
dpkg-query -W nvidia-l4t-core nvidia-jetpack nvidia-tensorrt 2>/dev/null
tegrastats --interval 1000 --logfile /tmp/safewatch_tegrastats.log
```

The repository now also provides a machine-readable target compatibility probe before the longer
measurement. Run it on the board from the project root; it records the JetPack identity, available
ONNX Runtime providers, RAM, model I/O contract and a finite-output smoke test of the exact files:

```bash
python3 scripts/edge_target_audit.py \
  --head export/onnx/2026-10-03_p3live_minirgbmel_fp32.onnx \
  --config runs/2026-10-03_p3live_minirgbmel/config.json
```

Then run the full benchmark while `tegrastats` records the process. Do not substitute a
VideoSwin/VGGish/CLIP checkpoint: only the MiniRGB + log-mel checkpoint has the feature contract
that the live extractor can reproduce.

`tegrastats` should run during the full-pipeline benchmark; stop it with Ctrl+C after the test. The
target's JetPack and available runtime versions are not visible from this x86 workspace.

## Supporting artifacts

- Edge head export/timing: `export/onnx/2026-10-03_p3live_minirgbmel_report.json`
- Host video-path budget: `export/onnx/2026-10-03_p3live_minirgbmel_edge_host_budget.json`
- MiniRGB export and distillation fidelity: `export/onnx/edge_budget_report.json`,
  `export/onnx/edge_distill_report.json`, and `export/onnx/edge_distill_train_report.json`
