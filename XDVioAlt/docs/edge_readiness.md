# Edge readiness: binary checkpoint

Checkpoint assessed: `runs/2026-10-08_ucf_abuse_sourceval_baseline_s43/ckpt_best.pt`.
The binary decision threshold remains **0.5**, as in the preceding evaluation. It was not adjusted
using the test split. This is a host-side benchmark and compatibility audit; it is not a measurement
on a Jetson, Raspberry Pi, or mini-PC deployment target.

## Head export benchmark

The model head was exported to ONNX and timed with ONNX Runtime's CPU provider on an AMD Ryzen 5
8640U host, 4 intra-op threads, using a synthetic 100-snippet input window. FP32 latency was
measured across 100 runs. FP16 and INT8 were also measured using the same setup.

| Export | File size | p50 / p95 latency per 100 snippets | Parity result |
|---|---:|---:|---|
| FP32 | 4.089 MB | 1.816 / 2.247 ms | Max clip probability difference `5.1e-7` on validation features; 0 threshold flips |
| FP16 | 2.079 MB | 3.775 / 10.894 ms | Max clip probability difference 0.0015; 0 threshold flips on 672 validation clips |
| INT8 per channel | 1.409 MB | 6.573 / 8.667 ms | Rejected: export snippet-logit drift 0.0575 exceeds 0.05 tolerance; max clip-probability difference 0.108 |

INT8 was slower than FP32 on this x86 host and did not pass the export parity tolerance. FP16 cut
the file size roughly in half but was also slower on this CPU; its speed on an ARM accelerator could
differ and has not been measured. FP32 and FP16 kept the same 0.5 decisions on these validation
examples, but this alone is not a deployment sign-off.

The measured process peak RSS was about 508 MiB for the validation parity process. This includes
Python, PyTorch, ONNX Runtime, loaded datasets and all three ONNX sessions; it is not a target-device
minimum RAM estimate.

## Compatibility blockers

The ONNX export declares a dynamic time axis, but cross-attention reshape operations were traced at
100 snippets. Direct inference on shorter sequences fails. This audit used uniformly sampled or
zero-padded 100-snippet validation windows and confirmed parity for the binary clip head on all 672
validation clips. Long-video chunking and aggregation are not yet verified for this checkpoint.

The live Edge extractor in the repository is MiniRGB distilled from I3D features. The assessed
checkpoint was trained with 1024-dimensional CLIP ViT-B-32 mean-plus-standard-deviation features.
The dimensions happen to match, but the feature spaces do not; feeding MiniRGB output to this head
is not validated and should not be treated as compatible. Therefore these numbers describe only the
classifier head on precomputed features. End-to-end raw-video latency, audio preprocessing latency,
and end-to-end accuracy for this checkpoint are still unknown.

The live inference function now fails closed when a raw video is paired with an unsupported
checkpoint, rather than generating a potentially misleading score. Benchmark clips with matching
precomputed features continue to use those exact features and remain reviewable.

## Artifacts

- ONNX export and synthetic-window timing: `export/onnx/2026-10-08_ucf_abuse_sourceval_baseline_s43_report.json`
- Validation clip-head parity and host RSS: `export/onnx/2026-10-08_ucf_abuse_sourceval_baseline_s43_edge_readiness.json`
- FP32, FP16 and INT8 ONNX files are in `export/onnx/` with the checkpoint run name.

## Readiness status

**The binary head exports cleanly, but this checkpoint is not end-to-end Edge-ready.** Before
deployment, either distill/train an extractor whose output matches this checkpoint's CLIP feature
space, or retrain the head on the existing MiniRGB representation; then validate variable-length
streaming/window aggregation and benchmark the full pipeline on the actual target hardware.

## MiniRGB-aligned binary checkpoint

A separate checkpoint, `runs/2026-10-26_binary_minirgb_fullval_s43/ckpt_best.pt`, was trained
on MiniRGB visual features and log-mel audio, using a source-held-out XD train/validation split.
Its feature contract matches the repository's live MiniRGB/log-mel extractor, so the app now offers
it first for live uploads and refuses raw-video inference with incompatible checkpoints. The split
builder is `scripts/prepare_minirgb_sourceholdout.py`; it reuses the existing cache and extracts only
the four missing clips. The checkpoint was selected at epoch 11 (30 epochs total), using
full-length validation clips so checkpoint selection matches the final evaluator and UI.

On the previously used official test set (799 clips; not a pristine test), the binary clip head
scored AP 0.900, ROC AUC 0.864, accuracy 0.735, precision 0.914, recall 0.636, and F1 0.750 at the
fixed 0.5 threshold. This is better than the prior MiniRGB checkpoint on the same list (AP 0.887,
recall 0.408), but below the latest CLIP checkpoint, which is not live-compatible. The auxiliary
abuse AP was 0.031 on only 11 test clips. Temporal localization remains weak: F1 at IoU 0.5 was
0.070. This solves feature-space compatibility, not abuse reliability, localization quality, or
Jetson readiness.
