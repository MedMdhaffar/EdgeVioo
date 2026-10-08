#!/usr/bin/env bash
# P3 (calibration fix) — does per-clip audio normalisation remove the global-AP penalty?
#
# Diagnosis (`docs/journal.md`, Session 9): the audio stream ranks well *inside* a clip but its score
# levels are not comparable *between* clips (audio-only: global AP 0.446 against per-clip 0.625), and
# removing audio even *raised* early fusion's global AP (0.8015 -> 0.8138). That is a gain/offset
# problem, not a capacity problem, so the candidate fix is instance normalisation of the audio
# channels over each clip (`MultimodalClipDataset(audio_norm="clip")`, affine-invariant per channel).
#
# Three runs isolate it:
#   p2_audio_full_clipnorm   audio alone      -> does the global/per-clip gap close?
#   p3full_late_clipnorm     late fusion      -> does fusion now beat visual-only on BOTH protocols?
#   p3full_adaptive_clipnorm adaptive + gate + dropout 0.15 -> the full proposed configuration
#
# Queues behind a running ladder (WAIT_FOR_LADDER=1) so it can be launched unattended.
# Usage:
#   WAIT_FOR_LADDER=1 setsid nohup bash scripts/run_p3_clipnorm.sh > data/logs/p3_clipnorm.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
THREADS="${THREADS:-6}"
EPOCHS="${EPOCHS:-30}"
LISTS="${LISTS:-data/lists}"
DROP_RATE="${DROP_RATE:-0.15}"
FORCE="${FORCE:-0}"

if [ "${WAIT_FOR_LADDER:-1}" = "1" ]; then
  sleep 10                 # grace period: a ladder started in the same second must be seen
  while pgrep -f "run_p3_full.sh" >/dev/null || pgrep -f "safewatch.train" >/dev/null; do
    echo "[clipnorm] $(date -Iseconds) waiting for the running ladder to finish"
    sleep 120
  done
fi

latest_run() { ls -dt runs/*_"$1" 2>/dev/null | head -1; }

# $1 = checkpoint, $2 = label, $3 = space-separated modalities that may be dropped
score_all() {
  local ckpt="$1" label="$2" drops="${3:-}"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" \
    || echo "[clipnorm] FAILED scoring ${label}"
  for modality in $drops; do
    echo "[clipnorm] scoring ${label} with ${modality} removed"
    "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" \
      --drop-modality "$modality" || echo "[clipnorm] FAILED ${label} (-${modality})"
  done
}

echo "[clipnorm] start $(date -Iseconds) lists=$LISTS"

# ------------------------------------------------------------------ audio alone (unimodal, calib.)
run=$(latest_run p2_audio_full_clipnorm)
if [ -f "$run/ckpt_best.pt" ] && [ "$FORCE" != "1" ]; then
  echo "[clipnorm] audio-only+clipnorm already trained -> $run"
else
  "$PY" -u -m safewatch.train --tag p2_audio_full_clipnorm --modality audio --audio-norm clip \
    --lists-dir "$LISTS" --epochs "$EPOCHS" --batch-size 32 --workers 4 --threads "$THREADS" \
    || echo "[clipnorm] FAILED audio-only+clipnorm"
  run=$(latest_run p2_audio_full_clipnorm)
fi
[ -f "$run/ckpt_best.pt" ] && score_all "$run/ckpt_best.pt" "audio-only+clipnorm" "audio"

# --------------------------------------------------------------------------- fusion rungs
for mode in late adaptive; do
  tag="p3full_${mode}_clipnorm"
  run=$(latest_run "$tag")
  if [ -f "$run/ckpt_best.pt" ] && [ "$FORCE" != "1" ]; then
    echo "[clipnorm] ${mode}+clipnorm already trained -> $run"
  else
    echo "[clipnorm] $(date -Iseconds) training fusion=${mode} audio_norm=clip"
    "$PY" -u -m safewatch.train_fusion --fusion-mode "$mode" --tag "$tag" --lists-dir "$LISTS" \
      --audio-norm clip --epochs "$EPOCHS" --modality-dropout "$DROP_RATE" \
      --workers 4 --threads "$THREADS" || echo "[clipnorm] FAILED training ${mode}"
    run=$(latest_run "$tag")
  fi
  [ -f "$run/ckpt_best.pt" ] && score_all "$run/ckpt_best.pt" "${mode}+clipnorm" "audio visual"
done

echo "[clipnorm] done $(date -Iseconds) - table: .venv/bin/python scripts/fusion_table.py clipnorm"
