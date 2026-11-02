#!/usr/bin/env bash
# Re-train the one-frame binary model, but select its checkpoint using full-length validation clips
# (batch size 1) exactly as final evaluation and the UI score them.  This removes the previous
# 200-snippet validation-selection mismatch.
#
# Run from the repository root:
#   bash scripts/run_binary_fullval_selection.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH="."

LISTS="data/lists_binary_minirgb_sourceholdout"
TAG="binary_minirgb_fullval_s43"
RUN="runs/$(date +%F)_${TAG}"
CKPT="$RUN/ckpt_best.pt"

echo "[full-val] train with full-length validation selection $(date +%H:%M:%S)"
"$PY" -u -m safewatch.train_fusion --fusion-mode cross --tag "$TAG" --seed 43 \
    --feature-set i3d_official --visual-dim 1024 --audio-dim 132 \
    --audio-dir data/features/mel --audio-stats data/features/audio_stats.json --audio-norm clip \
    --lists-dir "$LISTS" --epochs 30 --batch-size 8 --workers 0 --threads 4 \
    --class-over-sample auto --validation-max-snippets 0

echo "[full-val] select 94% recall threshold on validation only $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$LISTS/val.csv" \
    --select-recall-threshold 0.94

echo "[full-val] clean source-disjoint test at frozen validation threshold"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$LISTS/test.csv" \
    --threshold-from-json "$RUN/binary_validation_recall_threshold_selection_metrics.json" \
    --source-disjoint-from "$LISTS/train.csv" "$LISTS/val.csv" \
    --report-label recall94_validation
