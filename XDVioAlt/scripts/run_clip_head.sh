#!/usr/bin/env bash
# Step 1 of the CLIP roadmap — swap the visual features, keep everything else identical.
#
# The controlled experiment: `runs/2026-09-29_p3full_cross` (test AP 0.798) trains the
# reliability-gated cross-attention head on VideoSwin RGB features. Here the *only* changed variable is
# the visual stream, replaced by a frozen CLIP ViT-B/32 (data/features/clip). Same lists grid, same
# audio (`data/features/mel`), same hyper-parameters, same split — so the AP delta is attributable to
# the features and nothing else.
#
# Chain:  extract CLIP features -> train head -> score test (both / drop-audio / drop-video)
#
# Run: bash scripts/run_clip_head.sh                       # step 1 (1 frame/snippet)
#      K=4 POOL=mean+std bash scripts/run_clip_head.sh     # step 1b (temporal aggregation)
#      (unattended) setsid nohup bash scripts/run_clip_head.sh > data/logs/clip_head.log 2>&1 &
#
# Step 1b (K>1) is the follow-up to step 1's negative result: a single frame per snippet discarded the
# within-window motion, and the AP loss landed exactly on the motion classes. K frames per snippet,
# pooled with mean+std, turn that variation into an explicit cue. Cost scales with K.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH="."
THREADS="${THREADS:-8}"
EPOCHS="${EPOCHS:-30}"
K="${K:-1}"                 # frames per snippet (step 1b); 1 = the step-1 setup
POOL="${POOL:-mean}"        # "mean" -> 512-d, "mean+std" -> 1024-d (temporal variation)
if [ "$K" = "1" ] && [ "$POOL" = "mean" ]; then
  SUF=""
else
  SUF="_k${K}"
  if [ "$POOL" = "mean+std" ]; then SUF="${SUF}_mstd"; fi
fi
LISTS="${LISTS:-data/lists_clip${SUF}}"
if [ "$POOL" = "mean+std" ]; then VIS_DIM=1024; else VIS_DIM=512; fi
TAG="${TAG:-p3clip_cross${SUF}}"

# --- 1. features (resumable; skips clips already extracted) -------------------
if [ "${SKIP_EXTRACT:-0}" != "1" ]; then
  echo "[clip-head] $(date -Iseconds) extracting CLIP features (train val test, K=$K pool=$POOL)"
  "$PY" -u scripts/extract_clip.py --splits train val test --threads "$THREADS" \
    --workers 1 --batch-size 32 --frames-per-snippet "$K" --pool "$POOL" \
    || { echo "[clip-head] extraction failed"; exit 1; }
  echo "[clip-head] $(date -Iseconds) features ready -> $LISTS"
fi

# --- 2. train the head on CLIP visual + log-mel audio -------------------------
# feature_set stays "swin_rgb" on purpose: it selects the *stride* (64 frames/snippet) that the audio
# pooling and the list T values were built against; the visual *width* is what changes (512).
run=$(ls -dt runs/*_"$TAG" 2>/dev/null | head -1)
if [ -f "$run/ckpt_best.pt" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[clip-head] already trained -> $run"
else
  echo "[clip-head] $(date -Iseconds) training head on CLIP visual ($VIS_DIM-d) + mel audio"
  "$PY" -u -m safewatch.train_fusion --fusion-mode cross --tag "$TAG" \
    --lists-dir "$LISTS" --visual-dim "$VIS_DIM" --feature-set swin_rgb \
    --audio-dir data/features/mel --audio-stats data/features/audio_stats.json \
    --audio-norm none --epochs "$EPOCHS" --batch-size 32 --workers 4 --threads "$THREADS" \
    --seed 42 \
    || { echo "[clip-head] training failed"; exit 1; }
  run=$(ls -dt runs/*_"$TAG" 2>/dev/null | head -1)
fi

# --- 3. score the test split, and under a dropped modality (the sujet's robustness ask) ----------
ckpt="$run/ckpt_best.pt"
[ -f "$ckpt" ] || { echo "[clip-head] no checkpoint at $ckpt"; exit 1; }
echo "[clip-head] $(date -Iseconds) scoring ${run}"
"$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" \
  || echo "[clip-head] FAILED scoring"
"$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" --drop-modality audio \
  || echo "[clip-head] FAILED scoring (drop audio)"
"$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" --drop-modality visual \
  || echo "[clip-head] FAILED scoring (drop video)"

echo "[clip-head] $(date -Iseconds) comparison vs the VideoSwin baseline"
"$PY" - <<'PY'
import json
import pathlib

base = pathlib.Path("runs/2026-09-29_p3full_cross/test_metrics.json")
runs = sorted(pathlib.Path("runs").glob("*p3clip_cross*/test_metrics.json"))
print(f"{'run':34s} {'AP global':>9s} {'AP per-clip':>11s} {'F1@IoU.5':>8s}")
print("-" * 66)
if base.exists():
    m = json.loads(base.read_text())
    print(f"{'BASELINE (VideoSwin)':34s} {m['ap']['global__pr_auc']:9.4f} "
          f"{m['ap']['per_video__pr_auc']:11.4f} {m['localization']['f1@iou0.5']:8.4f}")
for path in runs:
    m = json.loads(path.read_text())
    print(f"{path.parent.name:34s} {m['ap']['global__pr_auc']:9.4f} "
          f"{m['ap']['per_video__pr_auc']:11.4f} {m['localization']['f1@iou0.5']:8.4f}")
PY
echo "[clip-head] ALL_DONE $(date -Iseconds)"
