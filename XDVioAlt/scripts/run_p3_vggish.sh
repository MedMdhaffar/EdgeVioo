#!/usr/bin/env bash
# P3-VGGISH — fusion + unimodal baselines on the OFFICIAL grid (I3D 1024-d visual
# + official VGGish 128-d audio, lists_official). Answers: does real VGGish audio
# beat the log-mel plateau (late 0.8125 global / 0.7825 per-clip on mirror grid)?
#
# Usage:
#   bash scripts/run_p3_vggish.sh              # smoke only (2 epochs, validates plumbing)
#   EPOCHS=30 bash scripts/run_p3_vggish.sh    # full ladder (hours on CPU)
#   (unattended) setsid nohup bash scripts/run_p3_vggish.sh > data/logs/p3_vggish.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
THREADS="${THREADS:-4}"
EPOCHS="${EPOCHS:-2}"
SUFFIX="${SUFFIX:-}"   # e.g. SUFFIX=30ep -> tags p2vggish30ep_visual (keeps smoke evidence intact)
LISTS="data/lists_official"
AUDIO_DIR="data/features/vggish_snippet"
AUDIO_STATS="data/features/vggish_stats.json"
QUALITY_STATS="data/features/quality_stats_official.json"
FUSION_BASE="--lists-dir $LISTS --feature-set i3d_official --visual-dim 1024 --audio-dir $AUDIO_DIR --audio-stats $AUDIO_STATS --audio-dim 128 --audio-grid snippet --quality-dir data/features/audio_quality --quality-stats $QUALITY_STATS --workers 0 --threads $THREADS"
UNI_BASE="--lists-dir $LISTS --feature-set i3d_official --audio-dir $AUDIO_DIR --audio-stats $AUDIO_STATS --audio-dim 128 --audio-grid snippet --workers 0 --threads $THREADS"
FORCE="${FORCE:-0}"
# Segment rule in SECONDS (2 snippets x 64 frames / 24 fps = the mirror-grid reference rule).
# Expressing it in snippets silently gives a 4x finer rule on this 16-frame grid (1.33 s), which
# wrecked the first localisation read-out of these runs (docs/journal.md Session 10).
MIN_SECONDS="${MIN_SECONDS:-5.333}"
MAX_GAP_SECONDS="${MAX_GAP_SECONDS:-2.667}"
latest_run() { ls -dt runs/*_"$1" 2>/dev/null | head -1; }
score_all() {
  local ckpt="$1" label="$2"
  local rule="--min-seconds $MIN_SECONDS --max-gap-seconds $MAX_GAP_SECONDS"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" $rule || echo "[vggish] FAILED scoring $label"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" $rule --drop-modality audio || echo "[vggish] FAILED drop-audio $label"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" $rule --drop-modality visual || echo "[vggish] FAILED drop-visual $label"
  "$PY" -u scripts/per_class_localization.py --predictions "$(dirname "$ckpt")/predictions.npz" --out "$(dirname "$ckpt")/per_class_localization.json" || echo "[vggish] FAILED per-class $label"
}
run_fusion() {
  local mode="$1" tag="p3vggish${SUFFIX}_${mode}"
  local run; run=$(latest_run "$tag")
  if [ -f "$run/ckpt_best.pt" ] && [ "$FORCE" != "1" ]; then echo "[vggish] $mode already trained -> $run";
  else
    echo "[vggish] $(date -Iseconds) training fusion=$mode epochs=$EPOCHS"
    # shellcheck disable=SC2086
    "$PY" -u -m safewatch.train_fusion --fusion-mode "$mode" --tag "$tag" $FUSION_BASE --epochs "$EPOCHS" --batch-size 8 || echo "[vggish] FAILED training $mode"
    run=$(latest_run "$tag")
  fi
  [ -f "$run/ckpt_best.pt" ] && score_all "$run/ckpt_best.pt" "$mode($run)"
}
echo "[vggish] start $(date -Iseconds) epochs=$EPOCHS"
if [ "${SKIP_UNIMODAL:-0}" != "1" ]; then
  for mod in visual audio; do
    tag="p2vggish${SUFFIX}_${mod}"; run=$(latest_run "$tag")
    if [ -f "$run/ckpt_best.pt" ] && [ "$FORCE" != "1" ]; then echo "[vggish] unimodal $mod already trained -> $run";
    else
      echo "[vggish] $(date -Iseconds) training unimodal $mod"
      # shellcheck disable=SC2086
      "$PY" -u -m safewatch.train --tag "$tag" --modality "$mod" $UNI_BASE --epochs "$EPOCHS" --batch-size 8 || echo "[vggish] FAILED unimodal $mod"
      run=$(latest_run "$tag")
    fi
    [ -f "$run/ckpt_best.pt" ] && "$PY" -u -m safewatch.eval.runner --ckpt "$run/ckpt_best.pt" --threads "$THREADS" || echo "[vggish] FAILED scoring $mod"
  done
fi
for mode in late cross adaptive early; do run_fusion "$mode"; done
echo "[vggish] done $(date -Iseconds)"; "$PY" scripts/fusion_table.py "p3vggish${SUFFIX}"; "$PY" scripts/fusion_table.py "p2vggish${SUFFIX}"
