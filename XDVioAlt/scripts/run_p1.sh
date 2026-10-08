#!/usr/bin/env bash
# P1 end-to-end run: wait for the feature download, build the split lists, train the visual-only
# MIL baseline and score the test split.
#
# Usage:
#   bash scripts/run_p1.sh
#   (unattended) setsid nohup bash scripts/run_p1.sh > /tmp/p1_run.log 2>&1 < /dev/null &
#
# Everything it produces is reproducible from docs/journal.md + runs/<date>_<tag>/config.json.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
FEATURE_SET="swin_rgb"
TAG="p1_visual_swin"
TARGET_FILES=4750          # 3 950 train clips + 800 test clips (one .npy per clip)
MAX_WAIT_MIN=90

echo "[p1] start $(date -Iseconds)"
waited=0
while true; do
  n=$(find "data/raw/xd-violence/data/${FEATURE_SET}" -name '*.npy' 2>/dev/null | wc -l)
  if [ "$n" -ge "$TARGET_FILES" ]; then
    echo "[p1] features complete: ${n}/${TARGET_FILES}"
    break
  fi
  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    echo "[p1] ABORT: only ${n}/${TARGET_FILES} feature files after ${waited} min"
    exit 1
  fi
  sleep 60
  waited=$((waited + 1))
done

echo "[p1] build split lists"
"$PY" scripts/build_lists.py --feature-set "$FEATURE_SET" --val-size 400

echo "[p1] train visual-only MIL baseline"
"$PY" -m safewatch.train --tag "$TAG" --feature-set "$FEATURE_SET" \
  --epochs 30 --batch-size 32 --workers 4 --threads 8

CKPT="$(ls -dt runs/*_"${TAG}" | head -1)/ckpt_best.pt"
echo "[p1] score test split with ${CKPT}"
"$PY" -m safewatch.eval.runner --ckpt "$CKPT" --stride-frames 64 --threads 8

echo "[p1] done $(date -Iseconds)"
