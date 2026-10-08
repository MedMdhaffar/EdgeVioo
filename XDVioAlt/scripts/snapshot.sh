#!/usr/bin/env bash
# Code snapshot — there is no Git history by choice (see README / docs/journal.md), so a small tar of
# the code, docs, split lists and annotations is the safety net against a lost or corrupted file.
# runs/ and data/features are deliberately excluded (huge; the per-run folders and the journal record
# what matters).
#
# Usage:
#   bash scripts/snapshot.sh                    # keeps the 10 most recent snapshots
#   DEST=/media/usb/sw KEEP=20 bash scripts/snapshot.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${DEST:-$HOME/safewatch_snapshots}"
KEEP="${KEEP:-10}"
NAME="safewatch_code_$(date +%Y-%m-%d_%H%M%S).tar.gz"
mkdir -p "$DEST"

tar czf "$DEST/$NAME" \
  safewatch tests scripts docs notebooks \
  README.md pyproject.toml requirements.txt requirements-lock.txt \
  data/lists data/lists_partial data/lists_official data/annotations

echo "[snapshot] $DEST/$NAME ($(du -h "$DEST/$NAME" | cut -f1))"
ls -1t "$DEST"/safewatch_code_*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do
  echo "[snapshot] pruning $(basename "$old")"
  rm -f "$old"
done
