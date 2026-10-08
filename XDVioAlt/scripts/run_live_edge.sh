#!/usr/bin/env bash
# End-to-end rebuild of the live/edge model so the import path stops being wrong.
#
# The chain, in order (each stage needs the previous one's output):
#   1. wait for scripts/edge_distill.py to finish -> minirgb_w24_distilled.pt (trained extractor)
#   2. re-export the ONNX extractor FROM those weights (replaces the random-init ONNX the live path
#      had been using, which is why uploads were scoring noise)
#   3. extract MiniRGB features for train/val/test -> the SAME representation inference produces
#   4. train the fusion head on (MiniRGB visual + log-mel audio)
#   5. score the test split with the run's own metric path
#
# Run: bash scripts/run_live_edge.sh   (long: ~4-5 h on CPU, so it is meant to run unattended)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH="."

while pgrep -f "scripts/edge_distill.py" >/dev/null; do sleep 30; done
echo "[chain] distill done $(date +%H:%M:%S)"

"$PY" -u scripts/edge_budget.py --extractor-weights export/onnx/minirgb_w24_distilled.pt \
    --videos 5 > data/logs/edge_budget_trained.log 2>&1
echo "[chain] onnx re-exported from distilled weights $(date +%H:%M:%S)"

"$PY" -u scripts/extract_minirgb.py --threads 1 --splits train val test \
    > data/logs/extract_minirgb.log 2>&1
echo "[chain] minirgb features extracted $(date +%H:%M:%S)"

"$PY" -u -m safewatch.train_fusion --fusion-mode cross --tag p3live_minirgbmel \
    --feature-set i3d_official --visual-dim 1024 --audio-dim 132 \
    --audio-dir data/features/mel --audio-stats data/features/audio_stats.json \
    --audio-norm clip --lists-dir data/lists_minirgb --epochs 30 \
    > data/logs/p3live_minirgbmel.log 2>&1
echo "[chain] head trained $(date +%H:%M:%S)"

RUN="runs/$(date +%F)_p3live_minirgbmel/ckpt_best.pt"
"$PY" -u -m safewatch.eval.runner --ckpt "$RUN" > data/logs/p3live_minirgbmel_score.log 2>&1
echo "[chain] scored $(date +%H:%M:%S) ALL_DONE"