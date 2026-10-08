#!/usr/bin/env bash
# Catch-up queue: finish the missing `late` rung, then run the calibration experiment.
#
# Why: the first `late` training of the 2026-09-28 ladder died with
#   AttributeError: 'MultimodalClipDataset' object has no attribute 'audio_norm'
# because the library was edited *while* that job was running: Python 3.14 defaults multiprocessing to
# `forkserver`, so a DataLoader worker re-imports `safewatch.data.multimodal` from disk and therefore
# ran the new `_audio()` against a dataset object pickled by the old `__init__`. Nothing else was hurt,
# the late rung simply has no checkpoint yet. Both scripts are resumable, so this wrapper is just:
#
#   1. wait for the ladder that is currently running,
#   2. re-run scripts/run_p3_full.sh  (skips trained variants, trains `late`, re-scores all),
#   3. run scripts/run_p3_clipnorm.sh (per-clip audio normalisation experiment).
#
# Lesson recorded in docs/journal.md: never edit the package while a job is running - queue the edit.
#
# Usage (unattended, survives the shell):
#   setsid nohup bash scripts/run_p3_catchup.sh > data/logs/p3_catchup.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[catchup] start $(date -Iseconds)"
sleep 10
while pgrep -f "run_p3_full.sh" >/dev/null || pgrep -f "safewatch.train" >/dev/null; do
  echo "[catchup] $(date -Iseconds) waiting for the running ladder"
  sleep 120
done

echo "[catchup] $(date -Iseconds) re-running the ladder (fills the late rung, re-scores the rest)"
FORCE=0 bash scripts/run_p3_full.sh

echo "[catchup] $(date -Iseconds) running the calibration experiment"
WAIT_FOR_LADDER=0 bash scripts/run_p3_clipnorm.sh

echo "[catchup] done $(date -Iseconds) - tables: .venv/bin/python scripts/fusion_table.py"
