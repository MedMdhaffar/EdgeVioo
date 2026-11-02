#!/usr/bin/env bash
# Joint CCTV + XD-Violence binary model training.
# Incorporates XD-Violence + RWF-2000 + SCFD.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="/home/aarf101/projtutore/venv_py312/bin/python"
export PYTHONPATH="."

LISTS="data/lists_joint_cctv_xdviolence"
TAG="binary_minirgb_joint_cctv_s43"
RUN="runs/$(date +%F)_${TAG}"
CKPT="$RUN/ckpt_best.pt"

echo "[joint-cctv] Start training on Joint XD-Violence + RWF-2000 + SCFD $(date +%H:%M:%S)"
"$PY" -u -m safewatch.train_fusion --fusion-mode cross --tag "$TAG" --seed 43 \
    --feature-set i3d_official --visual-dim 1024 --audio-dim 132 \
    --audio-dir data/features/mel --audio-stats data/features/audio_stats.json --audio-norm clip \
    --lists-dir "$LISTS" --epochs 30 --batch-size 8 --workers 0 --threads 4 \
    --class-over-sample auto --validation-max-snippets 0

echo "[joint-cctv] Select 94% recall operating threshold on joint validation $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$LISTS/val.csv" \
    --select-recall-threshold 0.94

echo "[joint-cctv] Evaluate on clean XD-Violence test split $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$LISTS/test.csv" \
    --threshold-from-json "$RUN/binary_validation_recall_threshold_selection_metrics.json" \
    --report-label xd_violence_clean_test

echo "[joint-cctv] Evaluate on RWF-2000 CCTV validation set $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "data/lists_rwf2000_binary/val.csv" \
    --threshold-from-json "$RUN/binary_validation_recall_threshold_selection_metrics.json" \
    --report-label rwf2000_cctv_val

echo "[joint-cctv] Evaluate on SCFD CCTV validation set $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "data/lists_scfd_binary/val.csv" \
    --threshold-from-json "$RUN/binary_validation_recall_threshold_selection_metrics.json" \
    --report-label scfd_cctv_val

echo "[joint-cctv] Training and evaluation complete! Check results in $RUN"

