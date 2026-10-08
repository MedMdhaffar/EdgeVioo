#!/usr/bin/env bash
# Session-12 remedy matrix - the trainings that test the fixes for the documented honest negatives.
#
#   #1 adaptive gate instability -> gate_source=reliability (the gate sees only the 7 measured
#      reliability channels, no embedding memorisation). Trained on 3 seeds; the claim being
#      checked is that the clean-model AP spread collapses toward the other fusion modes.
#   #7 abuse data starvation -> class_over_sample=auto (WeightedRandomSampler row-boost for
#      rare-class clips, exposure not loss re-weighting). Trained on cross, the best mode, s43.
#
# Everything else is the official grid, verbatim from runs/2026-09-30_p3vggish30ep_cross_s43.
#
# Usage:
#   bash scripts/run_remedy_matrix.sh
#   THREADS=4 bash scripts/run_remedy_matrix.sh          # leave CPU to a concurrent job
#   IDS="adaptive_relgate_43 cross_osample_43" bash scripts/run_remedy_matrix.sh
#   (unattended) setsid nohup bash scripts/run_remedy_matrix.sh > data/logs/remedy_matrix.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=".venv/bin/python"
THREADS="${THREADS:-4}"
read -r -a IDS <<< "${IDS:-adaptive_relgate_42 adaptive_relgate_43 adaptive_relgate_44 cross_osample_43}"

GRID=(
  --lists-dir data/lists_official
  --visual-dim 1024 --audio-dim 128
  --feature-set i3d_official
  --audio-dir data/features/vggish_snippet
  --audio-stats data/features/vggish_stats.json
  --quality-dir data/features/audio_quality
  --quality-stats data/features/quality_stats_official.json
  --audio-norm none --audio-grid snippet
  --n-reliability 7
  --epochs 30 --batch-size 8 --lr 0.0001 --max-snippets 200
  --k 5 --w-attn 0.5 --w-multi 0.3
  --emb 128 --hidden 512 --dropout 0.6 --heads 4
  --workers 0 --lr-milestones 10 --lr-gamma 0.1
  --modality-dropout 0.15 --multi-pos-weight none
)

for id in "${IDS[@]}"; do
  case "$id" in
    adaptive_relgate_*|adaptive_relgate_s*)
      fusion="adaptive"; seed="${id#adaptive_relgate_}"; seed="${seed#s}"
      extra=(--gate-source reliability)
      tag="p3vggish30ep_adaptive_relgate_s${seed}"
      ;;
    cross_osample_*|cross_osample_s*)
      fusion="cross"; seed="${id#cross_osample_}"; seed="${seed#s}"
      extra=(--class-over-sample auto)
      tag="p3vggish30ep_cross_osample_s${seed}"
      ;;
    *)
      echo "[remedy] unknown id '$id' - expected adaptive_relgate_sNN | cross_osample_sNN"
      continue
      ;;
  esac
  echo "[remedy] $(date -Iseconds) training $tag (fusion=$fusion seed=$seed threads=$THREADS)"
  "$PY" -u -m safewatch.train_fusion --tag "$tag" --fusion-mode "$fusion" \
    "${GRID[@]}" --seed "$seed" --threads "$THREADS" "${extra[@]}" \
    || echo "[remedy] FAILED $tag"
done

echo "[remedy] $(date -Iseconds) done"
