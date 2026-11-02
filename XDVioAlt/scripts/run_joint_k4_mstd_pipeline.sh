#!/usr/bin/env bash
# Complete multi-frame (k=4, mean+std, 2048-d) pipeline:
# 1. Extract k4_mstd features for RWF-2000 and SCFD
# 2. Build joint k4_mstd lists (combining with pre-existing XD-Violence k4_mstd)
# 3. Train the 2048-d joint model
# 4. Calibrate operating threshold via Maximum F1
# 5. Evaluate on XD-Violence clean test, RWF-2000 CCTV val, and SCFD CCTV val
set -euo pipefail
cd "$(dirname "$0")/.."
PY="/home/aarf101/projtutore/venv_py312/bin/python"
export PYTHONPATH="."

echo "[k4-cctv] 1. Extracting multi-frame CCTV features $(date +%H:%M:%S)"
"$PY" -u scripts/extract_cctv_features_k4_mstd.py

echo "[k4-cctv] 2. Building joint k4 dataset lists $(date +%H:%M:%S)"
"$PY" -u scripts/build_joint_cctv_k4_lists.py

LISTS="data/lists_joint_cctv_xdviolence_k4_mstd"
TAG="binary_minirgb_joint_cctv_k4_mstd_s43"
RUN="runs/$(date +%F)_${TAG}"
CKPT="$RUN/ckpt_best.pt"

echo "[k4-cctv] 3. Training 2048-d joint fusion model $(date +%H:%M:%S)"
"$PY" -u -m safewatch.train_fusion --fusion-mode cross --tag "$TAG" --seed 43 \
    --feature-set i3d_official --visual-dim 2048 --audio-dim 132 \
    --audio-dir data/features/mel --audio-stats data/features/audio_stats.json --audio-norm clip \
    --lists-dir "$LISTS" --epochs 30 --batch-size 8 --workers 0 --threads 4 \
    --class-over-sample auto --validation-max-snippets 0

echo "[k4-cctv] 4. Calibrate threshold for maximum F1-Score $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$LISTS/val.csv" \
    --select-f1-threshold

echo "[k4-cctv] 5. Evaluate on XD-Violence clean test $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "$LISTS/test.csv" \
    --threshold-from-json "$RUN/binary_validation_threshold_selection_metrics.json" \
    --report-label xd_violence_clean_test

echo "[k4-cctv] 6. Evaluate on RWF-2000 CCTV val $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "data/lists_rwf2000_binary_k4_mstd/val.csv" \
    --threshold-from-json "$RUN/binary_validation_threshold_selection_metrics.json" \
    --report-label rwf2000_cctv_val

echo "[k4-cctv] 7. Evaluate on SCFD CCTV val $(date +%H:%M:%S)"
"$PY" -u scripts/eval_binary_classifier.py --ckpt "$CKPT" --test-csv "data/lists_scfd_binary_k4_mstd/val.csv" \
    --threshold-from-json "$RUN/binary_validation_threshold_selection_metrics.json" \
    --report-label scfd_cctv_val

echo "[k4-cctv] ALL COMPLETE! Results in $RUN $(date +%H:%M:%S)"

