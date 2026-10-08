#!/usr/bin/env bash
# P4 — causal (streaming) scoring of the reference models on the OFFICIAL grid.
#
# Why this script exists: every number in the report so far comes from an offline pass that sees the
# whole clip at once (attention pools over all T snippets). A deployed alert cannot: at snippet t it
# has only t snippets. `--causal-prefix-stride K` re-forwards strided prefixes and keeps only each
# prefix-end score, so snippet t never sees the future. This turns the report's "not a real-time
# system" caveat into measured streaming AP / F1 / delay numbers.
#
# Cost note (measured, CPU, 4 threads): the exact K=1 pass is ~182x the offline pass; K=8 is ~23x.
# `cross` alone was 224.7 s for the 800 test clips. All three refs together run in ~10 min.
#
# Usage:
#   bash scripts/run_causal_score.sh
#   K=4 THREADS=4 bash scripts/run_causal_score.sh
#   TAGS="p3vggish30ep_cross_s43 p2vggish30ep_visual_s43" bash scripts/run_causal_score.sh  # seed check
#   (unattended) setsid nohup bash scripts/run_causal_score.sh > data/logs/causal.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=".venv/bin/python"
THREADS="${THREADS:-4}"
K="${K:-8}"
# Which runs to score. Default = the three report references; override to check other runs (seeds).
read -r -a REFS <<< "${TAGS:-p3vggish30ep_cross p3vggish30ep_late p2vggish30ep_visual}"

echo "[causal] start $(date -Iseconds) K=$K threads=$THREADS tags=${REFS[*]}"
for tag in "${REFS[@]}"; do
  ckpt=$(ls -dt runs/*"_$tag"/ckpt_best.pt 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then
    echo "[causal] skip $tag (no checkpoint)"
    continue
  fi
  echo "[causal] $(date -Iseconds) scoring $tag ckpt=$ckpt"
  "$PY" -u -m safewatch.eval.runner --ckpt "$ckpt" --threads "$THREADS" \
    --causal-prefix-stride "$K" --causal-progress 200 \
    || echo "[causal] FAILED $tag"
done

echo "[causal] offline vs causal (test split, official grid)"
"$PY" - "$K" "${REFS[@]}" <<'PY'
import json
import pathlib
import sys

k = sys.argv[1]
tags = sys.argv[2:]
header = f"{'run':30s} {'mode':>8s} {'APglob':>8s} {'APclip':>8s} {'F1@.5':>7s} {'delay_s':>8s}"
print(header)
print("-" * len(header))
for tag in tags:
    runs = sorted(pathlib.Path("runs").glob(f"*_{tag}"))
    if not runs:
        continue
    run = runs[-1]
    for label, fname in (("offline", "test_metrics.json"),
                         (f"causal{k}", f"test_metrics_causal{k}.json")):
        path = run / fname
        if not path.exists():
            print(f"{run.name:30s} {label:>8s} {'-':>8s} {'-':>8s} {'-':>7s} {'-':>8s}")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        loc = m["localization"]
        print(f"{run.name:30s} {label:>8s} "
              f"{m['ap']['global__pr_auc']:8.4f} {m['ap']['per_video__average_precision']:8.4f} "
              f"{loc['f1@iou0.5']:7.4f} {loc['mean_detection_delay_seconds']:8.2f}")
PY
echo "[causal] done $(date -Iseconds)"
