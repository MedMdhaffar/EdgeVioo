#!/usr/bin/env bash
# One-glance project status.  Usage:  bash scripts/status.sh
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
LOG="data/logs/p2_audio.log"
TARGET=4749        # 4 750 clips minus the one corrupt file in the mirror

mel=$(find data/features/mel -name '*.npy' 2>/dev/null | wc -l)
vgg=$(find data/features/vggish -name '*.npy' 2>/dev/null | wc -l)
vids=$(find data/raw/xd-violence/data/video -name '*.mp4' 2>/dev/null | wc -l)
i3d=$(ls "$HOME"/xd_official/i3d-features/RGB/*.npy 2>/dev/null | wc -l)
chunks=$(grep -c 'done, feature files' "$LOG" 2>/dev/null || true)
skipped=$(cat data/features/skipped_*.json 2>/dev/null | grep -c '"file"' || true)

echo "audio features (mel)     : ${mel} / ~${TARGET}"
echo "audio features (vggish)  : ${vgg} (subset only)"
echo "videos on disk           : ${vids}"
echo "official I3D RGB crops   : ${i3d} (5 x 4 754 clips expected)"
echo "train chunks finished    : ${chunks} / 5"
echo "mirror files skipped     : ${skipped}"
echo "free disk                : $(df -h . | awk 'NR==2 {print $4}')"

job() {  # job <label> <pattern>
  if pgrep -f "$2" >/dev/null; then echo "$1: RUNNING"; else echo "$1: stopped"; fi
}
job "sweep job (full ladder)  " run_p3_full
job "calibration experiment   " run_p3_clipnorm
job "catch-up queue           " run_p3_catchup
job "scoring script           " run_p3_score
# any training process, listed with its actual command (the old fixed labels mislabelled whichever
# job happened to match "safewatch.train" - the newest run is what matters, not a guessed phase)
if pgrep -f "safewatch.train" >/dev/null; then
  echo "training now             :"
  pgrep -af "safewatch.train" | sed 's/^/  /' | cut -c1-150
else
  echo "training now             : (none)"
fi
if grep -q 'all done' "$LOG" 2>/dev/null; then echo "P2 result                : FINISHED"; fi

echo "--- runs (newest last) ---"
for run in runs/*/; do
  name=$(basename "$run")
  [ -f "${run}ckpt_best.pt" ] || { echo "  ${name}  (no checkpoint)"; continue; }
  "$PY" - "$run" <<'PY'
import json, pathlib, sys

run = pathlib.Path(sys.argv[1])
val = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else {}
test = run / "test_metrics.json"
if not test.exists():
    print(f"  {run.name:38s} val_ap={val.get('val_ap', float('nan')):.4f} test: NOT SCORED")
else:
    m = json.loads(test.read_text())
    drop = {}
    for suffix in ("audio", "visual"):
        path = run / f"test_metrics_drop_{suffix}.json"
        if path.exists():
            drop[suffix] = json.loads(path.read_text())["ap"]["global__pr_auc"]
    # label the protocol explicitly: pr_auc (Wu-compatible) and average_precision (tie-safe) diverge,
    # so a table that shows one under the name of the other is simply wrong
    print(f"  {run.name:38s} val_ap={val.get('val_ap', float('nan')):.4f} "
          f"global pr_auc={m['ap']['global__pr_auc']:.4f} "
          f"per-clip AP={m['ap']['per_video__average_precision']:.4f} "
          f"-audio={drop.get('audio', float('nan')):.4f} "
          f"-video={drop.get('visual', float('nan')):.4f} n={m['n_clips']}")
PY
done

echo "--- last log lines ---"
tail -3 "$LOG" 2>/dev/null || echo "(no log yet)"

