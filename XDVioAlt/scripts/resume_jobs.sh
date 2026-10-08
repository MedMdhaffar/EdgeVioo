#!/usr/bin/env bash
# Resume the experiment queue after an interruption (reboot, logout, closed terminal).
#
# Why this exists: the 2026-09-28 evening ladder was killed by a machine shutdown at 23:36 (the box
# came back at 07:09 with nothing running), which is easy to miss - the logs simply stop. Every script
# in this repo is resumable, so recovery is "re-launch the chain", and this is that one command.
#
# What it does: if no project job is alive, start scripts/run_p3_catchup.sh (ladder -> calibration
# experiment). If a job *is* alive it refuses, so it is always safe to call.
#
# Usage:
#   bash scripts/resume_jobs.sh
#   (from a cron/`@reboot` line, unattended)
set -uo pipefail
cd "$(dirname "$0")/.."

if pgrep -f "run_p3_catchup.sh|run_p3_full.sh|run_p3_clipnorm.sh|safewatch.train" >/dev/null; then
  echo "[resume] a job is already running - nothing to do"
  pgrep -af "safewatch.train" | cut -c1-120
  exit 0
fi

echo "[resume] nothing running: starting the catch-up chain $(date -Iseconds)"
setsid nohup bash scripts/run_p3_catchup.sh >> data/logs/p3_catchup.log 2>&1 &
sleep 5
echo "[resume] started (pid $!); follow it with: tail -f data/logs/p3_catchup.log"
echo "[resume] what is missing: bash scripts/status.sh"
