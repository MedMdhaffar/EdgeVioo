#!/usr/bin/env bash
# P3 — score the four fusion checkpoints on the **test** split.
#
# Why this script exists: the four 2026-09-24 fusion runs only had clip-level *validation* AP
# (0.988-0.993, saturated from epoch 1, so it cannot rank the variants). The test-set numbers were
# missing because the scorer could not rebuild a FusionModel at all. It can now.
#
# Each run is scored on the lists it was trained with (data/lists_partial -> the 799 test clips that
# have audio features) and with the grid inferred from its checkpoint. Results land in
# runs/<run>/test_metrics.json + predictions.npz; a comparison table is printed at the end.
#
# Usage:
#   bash scripts/run_p3_score.sh
#   WAIT_FOR_TRAIN=1 bash scripts/run_p3_score.sh        # queue behind a running training job
#   (unattended) setsid nohup bash scripts/run_p3_score.sh > data/logs/p3_score.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
THREADS="${THREADS:-6}"

if [ "${WAIT_FOR_TRAIN:-0}" = "1" ]; then
  sleep 10                      # grace period: a training job started in the same second must be seen
  while pgrep -f "safewatch.train " >/dev/null; do
    echo "[p3-score] $(date -Iseconds) waiting for the running training job"
    sleep 60
  done
fi

for run in runs/*_p3_early runs/*_p3_late runs/*_p3_cross runs/*_p3_adaptive; do
  ckpt="${run}/ckpt_best.pt"
  [ -f "$ckpt" ] || { echo "[p3-score] skip ${run} (no checkpoint)"; continue; }
  echo "[p3-score] $(date -Iseconds) scoring ${ckpt}"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" \
    || echo "[p3-score] FAILED ${ckpt}"
done

echo "[p3-score] comparison (test split)"
"$PY" - <<'PY'
import json
import pathlib

rows = []
for path in sorted(pathlib.Path("runs").glob("*_p3_*/test_metrics.json")):
    m = json.loads(path.read_text(encoding="utf-8"))
    rows.append((path.parent.name, m["n_clips"], m["ap"]["global__pr_auc"],
                 m["ap"]["per_video__pr_auc"], m["localization"]["f1@iou0.5"]))
header = f"{'run':34s} {'clips':>5s} {'AP global':>9s} {'AP per-clip':>11s} {'F1@IoU.5':>8s}"
print(header)
print("-" * len(header))
for name, n, glob, per, f1 in rows:
    print(f"{name:34s} {n:5d} {glob:9.4f} {per:11.4f} {f1:8.4f}")
PY
