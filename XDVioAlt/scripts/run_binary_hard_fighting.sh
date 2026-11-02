#!/usr/bin/env bash
# Train the deployed one-frame MiniRGB representation with extra exposure to *training-only*
# martial-arts/action fighting.  Held-out Ip Man 3 and Taken 3 sources are never used here.
#
# Run from the repository root:
#   bash scripts/run_binary_hard_fighting.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH="."

LISTS="data/lists_binary_minirgb_sourceholdout"
TAG="binary_minirgb_hardfighting_s43"
RUN="runs/$(date +%F)_${TAG}"
CKPT="$RUN/ckpt_best.pt"
HARD_SOURCES="IP.Man.2.2010,Ip.Man.2008,Taken.2.UNRATED.EXTENDED.2012"

echo "[hard-fighting] train one-frame binary head $(date +%H:%M:%S)"
"$PY" -u -m safewatch.train_fusion --fusion-mode cross --tag "$TAG" --seed 43 \
    --feature-set i3d_official --visual-dim 1024 --audio-dim 132 \
    --audio-dir data/features/mel --audio-stats data/features/audio_stats.json --audio-norm clip \
    --lists-dir "$LISTS" --epochs 30 --batch-size 8 --workers 0 --threads 4 \
    --class-over-sample auto --hard-fighting-source-terms "$HARD_SOURCES" \
    --hard-fighting-weight 3

echo "[hard-fighting] choose threshold on validation only $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$LISTS/val.csv" \
    --select-f1-threshold

echo "[hard-fighting] clean source-disjoint test at frozen validation threshold"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$LISTS/test.csv" \
    --threshold-from-json "$RUN/binary_validation_threshold_selection_metrics.json" \
    --source-disjoint-from "$LISTS/train.csv" "$LISTS/val.csv" \
    --report-label f1_validation_threshold
