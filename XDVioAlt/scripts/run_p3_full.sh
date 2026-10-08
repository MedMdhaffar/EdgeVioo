#!/usr/bin/env bash
# P2/P3 (FULL corpus) — the ablation ladder on the complete audio corpus, with the reliability-aware
# adaptive gate and modality-dropout training.
#
# Why this script exists
# ----------------------
# The four 2026-09-24 fusion runs were trained on `data/lists_partial` (1 746 train clips = 45 % of the
# train split) because the audio extraction was still running, and their "adaptive" gate was trained
# with `n_reliability=0` (blind to the signal-quality statistics). The audio corpus is now complete
# (4 749 / 4 749 mel features), so the whole ladder is re-measured here on `data/lists`
# (3 516 train / 434 val / 800 test clips):
#
#   audio-only  ->  early  ->  late  ->  cross  ->  adaptive(gate + 7 reliability channels)
#
# Each model is scored three times: both modalities, audio removed, video removed
# (`--drop-modality`), which is the subject's "rester fonctionnelle lorsqu'une modalite est degradee
# ou indisponible" turned into numbers instead of a claim.
#
# Resumable: a variant whose checkpoint exists is skipped (FORCE=1 re-runs everything).
#
# Usage:
#   bash scripts/run_p3_full.sh
#   (unattended) setsid nohup bash scripts/run_p3_full.sh > data/logs/p3_full.log 2>&1 &
set -uo pipefail            # NOT -e: one failed variant must not kill the whole ladder
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
THREADS="${THREADS:-6}"
EPOCHS="${EPOCHS:-30}"
LISTS="${LISTS:-data/lists}"
DROP_RATE="${DROP_RATE:-0.15}"
FORCE="${FORCE:-0}"

test -f "$LISTS/train.csv" || { echo "[p3full] missing $LISTS/train.csv"; exit 1; }
echo "[p3full] start $(date -Iseconds) lists=$LISTS epochs=$EPOCHS dropout=$DROP_RATE"
echo "[p3full] reliability stats: $(test -f data/features/quality_stats.json && echo present || echo MISSING)"

latest_run() { ls -dt runs/*_"$1" 2>/dev/null | head -1; }

score_all() {   # $1 = checkpoint, $2 = run label
  local ckpt="$1" label="$2"
  echo "[p3full] scoring ${label} (both / drop-audio / drop-video)"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" \
    || echo "[p3full] FAILED scoring ${label}"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" --drop-modality audio \
    || echo "[p3full] FAILED scoring ${label} (drop audio)"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" --drop-modality visual \
    || echo "[p3full] FAILED scoring ${label} (drop video)"
}

# --------------------------------------------------------------------- audio-only (unimodal rung)
run=$(latest_run p2_audio_full)
if [ -f "$run/ckpt_best.pt" ] && [ "$FORCE" != "1" ]; then
  echo "[p3full] audio-only already trained -> $run"
else
  echo "[p3full] $(date -Iseconds) training audio-only MIL on $LISTS"
  "$PY" -u -m safewatch.train --tag p2_audio_full --modality audio --lists-dir "$LISTS" \
    --epochs "$EPOCHS" --batch-size 32 --workers 4 --threads "$THREADS" \
    || echo "[p3full] FAILED audio-only training"
  run=$(latest_run p2_audio_full)
fi
[ -f "$run/ckpt_best.pt" ] && score_all "$run/ckpt_best.pt" "audio-only(${run##*/})"

# --------------------------------------------------------------------------- four fusion variants
for mode in early late cross adaptive; do
  tag="p3full_${mode}"
  run=$(latest_run "$tag")
  if [ -f "$run/ckpt_best.pt" ] && [ "$FORCE" != "1" ]; then
    echo "[p3full] ${mode} already trained -> $run"
  else
    echo "[p3full] $(date -Iseconds) training fusion=${mode} (reliability gate + dropout $DROP_RATE)"
    "$PY" -u -m safewatch.train_fusion --fusion-mode "$mode" --tag "$tag" --lists-dir "$LISTS" \
      --epochs "$EPOCHS" --modality-dropout "$DROP_RATE" --workers 4 --threads "$THREADS" \
      || echo "[p3full] FAILED training ${mode}"
    run=$(latest_run "$tag")
  fi
  [ -f "$run/ckpt_best.pt" ] && score_all "$run/ckpt_best.pt" "${mode}(${run##*/})"
done

echo "[p3full] done $(date -Iseconds) - run: bash scripts/status.sh for the table"
