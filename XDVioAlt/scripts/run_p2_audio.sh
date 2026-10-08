#!/usr/bin/env bash
# P2 — audio pipeline: download XD-Violence videos in chunks, extract log-mel + quality features,
# then delete the train videos to bound disk usage. The **test** videos are kept: they are needed later
# for the UI, key frames and spectrograms.
#
# Resumable *and* coverage-aware: a chunk is skipped when every clip of that chunk already has its
# feature file (the mirror's own chunked feature dirs in data/raw/.../swin_rgb/<chunk>/ are the
# reference list of which clips belong to a chunk). Without that check a restart re-downloads ~14 GB
# of videos whose features are already on disk - observed on 2026-09-25 and fixed here.
#
# Usage:
#   bash scripts/run_p2_audio.sh
#   (unattended, keeps the history) setsid nohup bash scripts/run_p2_audio.sh >> data/logs/p2_audio.log 2>&1 &
set -uo pipefail          # deliberately NOT -e: a flaky download must not abort a multi-hour job
cd "$(dirname "$0")/.."

PY=".venv/bin/python"           # call sites add -u for unbuffered logs
REPO="jherng/xd-violence"
ROOT="data/raw/xd-violence"
FEATURES="data/features/mel"
SWIN="data/raw/xd-violence/data/swin_rgb"
TRAIN_CHUNKS=(1-1004 1005-2004 2005-2804 2805-3319 3320-3954)

count() { find "$1" -name '*.mp4' 2>/dev/null | wc -l; }
features() { find "$FEATURES" -name '*.npy' 2>/dev/null | wc -l; }

chunk_clips() { ls "$SWIN/$1"/*.npy 2>/dev/null | wc -l; }        # clips that belong to the chunk
chunk_have() {                                                     # ... that have a feature file
  "$PY" - "$1" <<'PYEOF'
import pathlib, sys
base = pathlib.Path(f"data/raw/xd-violence/data/swin_rgb/{sys.argv[1]}")
mel = pathlib.Path("data/features/mel")
print(sum(1 for p in base.glob("*.npy") if (mel / f"{p.stem}.npy").exists()))
PYEOF
}

retry_download() {        # per-file retries + skip-on-failure (the hf CLI aborts whole batches)
  local prefix="$1"
  "$PY" -u scripts/robust_download.py --include-prefix "$prefix" --out "$ROOT" \
    --skip-log "data/features/skipped_$(echo "$prefix" | tr '/.' '__').json"
}

echo "[p2] start $(date -Iseconds)  free disk: $(df -h . | awk 'NR==2 {print $4}')"

# --------------------------------------------------------------------------- test videos (kept)
test_expected=$(ls "$SWIN/test_videos"/*.npy 2>/dev/null | wc -l)
test_have=$("$PY" - <<'PYEOF'
import csv, pathlib
rows = list(csv.DictReader(open("data/lists/test.csv", newline="", encoding="utf-8")))
mel = pathlib.Path("data/features/mel")
print(sum(1 for r in rows if (mel / f"{r['clip_id']}.npy").exists()))
PYEOF
)
missing_mirror=$(cat data/features/skipped_*test*.json 2>/dev/null | grep -c '"file"' || true)
if [ "$test_have" -ge $((test_expected - missing_mirror)) ] && [ "$test_expected" -gt 0 ]; then
  echo "[p2] test features already complete: ${test_have}/${test_expected} (mirror files skipped: ${missing_mirror})"
else
  if [ "$(count "${ROOT}/data/video/test_videos")" -lt 800 ]; then
    echo "[p2] downloading the 800 test videos (~14 GB, kept for the demo/UI)"
    retry_download "data/video/test_videos/"
  fi
  echo "[p2] test videos present: $(count "${ROOT}/data/video/test_videos")/800"
  echo "[p2] extracting audio features for test clips"
  "$PY" -u scripts/extract_audio_features.py --videos "${ROOT}/data/video/test_videos" \
    --log data/features/audio_log_test.jsonl
fi

# --------------------------------------------------------------------------- train videos (deleted)
for chunk in "${TRAIN_CHUNKS[@]}"; do
  total=$(chunk_clips "$chunk")
  have=$(chunk_have "$chunk")
  if [ "$total" -gt 0 ] && [ "$have" -ge "$total" ]; then
    echo "[p2] chunk ${chunk}: features already complete (${have}/${total}) -> skip"
    continue
  fi
  echo "[p2] chunk ${chunk}: ${have}/${total} features, $(count "${ROOT}/data/video/${chunk}") videos present"
  echo "[p2] downloading chunk ${chunk} (existing files are skipped, so a partial chunk resumes)"
  retry_download "data/video/${chunk}/"
  echo "[p2] extracting audio features for chunk ${chunk}"
  "$PY" -u scripts/extract_audio_features.py --videos "${ROOT}/data/video/${chunk}" \
    --log data/features/audio_log_train.jsonl
  echo "[p2] chunk ${chunk} extracted -> deleting videos to free disk"
  rm -rf "${ROOT}/data/video/${chunk}"
  echo "[p2] chunk ${chunk} done, feature files: $(features) (chunk coverage $(chunk_have "$chunk")/$(chunk_clips "$chunk")), free disk: $(df -h . | awk 'NR==2 {print $4}')"
done

echo "[p2] all done $(date -Iseconds)  feature files: $(features)"
du -sh data/features/vggish data/features/audio_quality 2>/dev/null || true

