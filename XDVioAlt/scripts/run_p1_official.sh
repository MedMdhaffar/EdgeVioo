#!/usr/bin/env bash
# P1 (official features) — train the visual-only MIL baseline on the **official** I3D RGB features
# and score the test split.
#
# The official features live outside the repo (~45 GB, ~/xd_official/i3d-features) and their grid is
# 16 frames/snippet, where the HF mirror's swin grid is 64. The scorer infers the grid from the
# checkpoint (safewatch/eval/runner.py::STRIDE_FRAMES), so no --stride-frames is needed here.
#
# Usage:
#   bash scripts/run_p1_official.sh
#   (unattended) setsid nohup bash scripts/run_p1_official.sh > data/logs/p1_official.log 2>&1 &
#
# History: the first attempt (2026-09-25 10:50) died on its first batch with
# `KeyError: 'visual'` — the dataset/loop key mismatch fixed in safewatch/train.py; the traceback is
# kept in data/logs/p1_official_crash_keyerror.log.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
TAG="p1_official_rgb"
LISTS="data/lists_official"
FEATURES="$HOME/xd_official/i3d-features/RGB"

test -f "$LISTS/train.csv" || { echo "[p1-official] missing $LISTS/train.csv (build it first)"; exit 1; }
echo "[p1-official] start $(date -Iseconds)"
echo "[p1-official] official feature files: $(ls "$FEATURES"/*.npy 2>/dev/null | wc -l) (5 crops x clips)"

echo "[p1-official] train visual-only MIL baseline (30 epochs)"
"$PY" -u -m safewatch.train --tag "$TAG" --feature-set i3d_official --lists-dir "$LISTS" \
  --modality visual --epochs 30 --batch-size 32 --workers 4 --threads 8

CKPT="$(ls -dt runs/*_"${TAG}" | head -1)/ckpt_best.pt"
echo "[p1-official] score test split with ${CKPT}"
"$PY" -u -m safewatch.eval.runner --ckpt "$CKPT" --threads 8

echo "[p1-official] done $(date -Iseconds)"
