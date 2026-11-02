#!/usr/bin/env bash
# Controlled visual-representation experiment for slow, karate-style fights missed by the current
# one-frame MiniRGB head.  It never overwrites the shipped one-frame features/checkpoint.
#
# Stages: 4 frames/snippet -> mean+std (2048-d) MiniRGB features -> binary fusion training ->
# validation threshold selection -> clean source-disjoint test evaluation.
#
# Run from the repository root:
#   nohup bash scripts/run_binary_minirgb_multiframe.sh > data/logs/minirgb_k4_mstd.log 2>&1 &
#   tail -f data/logs/minirgb_k4_mstd.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH="."

LISTS="data/lists_binary_minirgb_sourceholdout"
FEATURES="data/features/minirgb_k4_mstd"
FEATURE_LISTS="data/lists_binary_minirgb_sourceholdout_k4_mstd"
TAG="binary_minirgb_k4_mstd_s43"
RUN="runs/$(date +%F)_${TAG}"
CKPT="$RUN/ckpt_best.pt"

echo "[k4-mstd] extract features (resumable) $(date +%H:%M:%S)"
"$PY" -u scripts/extract_minirgb.py --lists "$LISTS" --splits train val test \
    --out "$FEATURES" --out-lists "$FEATURE_LISTS" --threads 1 \
    --frames-per-snippet 4 --pool mean+std

echo "[k4-mstd] train binary head $(date +%H:%M:%S)"
"$PY" -u -m safewatch.train_fusion --fusion-mode cross --tag "$TAG" --seed 43 \
    --feature-set i3d_official --visual-dim 2048 --audio-dim 132 \
    --audio-dir data/features/mel --audio-stats data/features/audio_stats.json --audio-norm clip \
    --lists-dir "$FEATURE_LISTS" --epochs 30 --batch-size 8 --workers 0 --threads 4

echo "[k4-mstd] choose operating threshold on validation only $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" \
    --test-csv "$FEATURE_LISTS/val.csv" --select-f1-threshold

echo "[k4-mstd] score clean source-disjoint test at the frozen validation threshold"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$FEATURE_LISTS/test.csv" \
    --threshold-from-json "$RUN/binary_validation_threshold_selection_metrics.json" \
    --source-disjoint-from "$FEATURE_LISTS/train.csv" "$FEATURE_LISTS/val.csv" \
    --report-label f1_validation_threshold
echo "[k4-mstd] ALL_DONE $(date +%H:%M:%S)"
