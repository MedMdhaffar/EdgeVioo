#!/usr/bin/env bash
# Finish the live/edge rebuild end to end. SAFE TO RE-RUN after any interruption or reboot:
#   * extraction is idempotent (it validates each cached .npy and re-does only what is missing/young)
#   * training overwrites its own run dir; scoring overwrites its own predictions
#
# Stages: extract MiniRGB features (train/val/test) -> train the fusion head on that representation
# (plus log-mel audio) -> score the test split.
#
#   cd /home/aarf101/projtutore
#   nohup bash scripts/run_live_edge_head.sh > data/logs/head_chain.log 2>&1 &
#   tail -f data/logs/head_chain.log          # follow
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH="."

echo "[live] extract features (resumable) $(date +%H:%M:%S)"
"$PY" -u scripts/extract_minirgb.py --threads 1 --splits train val test

echo "[live] train head on (MiniRGB + log-mel) $(date +%H:%M:%S)"
"$PY" -u -m safewatch.train_fusion --fusion-mode cross --tag p3live_minirgbmel \
    --feature-set i3d_official --visual-dim 1024 --audio-dim 132 \
    --audio-dir data/features/mel --audio-stats data/features/audio_stats.json \
    --audio-norm clip --lists-dir data/lists_minirgb --epochs 30

RUN="runs/$(date +%F)_p3live_minirgbmel/ckpt_best.pt"
echo "[live] score test with $RUN $(date +%H:%M:%S)"
"$PY" -u -m safewatch.eval.runner --ckpt "$RUN"
echo "[live] ALL_DONE $(date +%H:%M:%S)"
