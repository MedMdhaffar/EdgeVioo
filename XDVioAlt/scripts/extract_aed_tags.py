#!/usr/bin/env python
"""Extract YAMNet acoustic event tags for the clips of a split (frozen AED, no training).

Why
---
The subject requires naming the acoustic events behind an alert ("reconnaissance d'evenements
acoustiques"). The project's audio stream was previously described with signal statistics only
(rms, flatness, ...). This script runs the frozen YAMNet tagger (521 AudioSet-YouTube classes,
``safewatch.edge.aed``) over the raw audio of each clip and stores, per snippet of the *official*
grid, the mean tag probability of the overlapping 0.96 s YAMNet frames. Everything downstream
(``scripts/incident_sheet.py``, ``safewatch/explain``, the console) reads the resulting npz;
nothing is re-inferred on the benchmark.

Model artefacts (download once, ~16 MB, see ``docs/journal.md`` AED session):
    curl -L -o data/features/aed/yamnet.onnx \
        https://huggingface.co/jafet21/yamnetonnx/resolve/main/yamnet.onnx
    curl -L -o data/features/aed/yamnet_class_map.csv \
        https://huggingface.co/jafet21/yamnetonnx/resolve/main/yamnet_class_map.csv

Usage:
    .venv/bin/python scripts/extract_aed_tags.py --splits test            # ~10-20 min on the CPU
    .venv/bin/python scripts/extract_aed_tags.py --splits test train val
    .venv/bin/python scripts/extract_aed_tags.py --splits test --limit 10  # quick plausibility run

Outputs:
    data/features/aed/tags/<clip_id>.npz   {tags: (T, 521) float16}
    data/features/aed/aed_stats.json       coverage, timings, top classes, and - on the test split
                                           - the AP (global and per-clip) of the curated
                                           incident-energy score against the snippet ground truth
                                           (honest check, not a win)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.data.dataset import load_rows  # noqa: E402
from safewatch.edge import aed  # noqa: E402
from safewatch.edge.audio import decode_audio  # noqa: E402
from safewatch.eval import metrics as M  # noqa: E402

TAGS_DIR = aed.TAGS_DIR
VIDEO_ROOT = Path("data/raw/xd-violence/data/video")
SNIPPET_S = 16 / 24.0       # official grid: one snippet per 16 frames


def video_index() -> dict[str, Path]:
    return {p.stem: p for p in VIDEO_ROOT.rglob("*.mp4")}


def load_intervals() -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    path = Path("data/annotations/test_intervals.csv")
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out.setdefault(row["clip_id"], []).append((int(row["start"]), int(row["end"])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="stop after N clips (plausibility)")
    args = parser.parse_args()

    if not aed.yamnet_available():
        raise SystemExit("YAMNet artefacts missing - download them (docstring above) first")

    videos = video_index()
    intervals = load_intervals()
    idx, _ = aed.incident_tag_indices()
    names = aed.tag_names()
    TAGS_DIR.mkdir(parents=True, exist_ok=True)

    class_hits = Counter()
    no_audio = []
    t_start = time.perf_counter()
    n_done = 0

    for split in args.splits:
        rows = load_rows(f"data/lists_official/{split}.csv")
        print(f"[{split}] {len(rows)} clips")
        for row in rows:
            if args.limit and n_done >= args.limit:
                break
            clip_id = row["clip_id"]
            out = TAGS_DIR / f"{clip_id}.npz"
            if out.exists():
                continue
            video = videos.get(clip_id)
            if video is None:
                no_audio.append(clip_id)
                continue
            audio = decode_audio(video)
            if audio.size == 0:
                no_audio.append(clip_id)
                continue
            frames = aed.yamnet_frames(audio)
            tags = aed.snippet_tags(frames, int(row["T"]), snippet_s=SNIPPET_S)
            np.savez_compressed(out, tags=tags.astype(np.float16))
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  {n_done} done, {elapsed / n_done:.2f} s/clip")
        if args.limit and n_done >= args.limit:
            break

    # --- audit, computed from the on-disk state so a resume/re-run yields identical stats ---
    on_disk = {p.stem for p in TAGS_DIR.glob("*.npz")}
    n_written = 0
    ap_scores: dict[str, np.ndarray] = {}
    ap_labels: dict[str, np.ndarray] = {}
    for split in args.splits:
        for row in load_rows(f"data/lists_official/{split}.csv"):
            clip_id = row["clip_id"]
            if clip_id not in on_disk:
                continue
            tags = np.asarray(np.load(TAGS_DIR / f"{clip_id}.npz")["tags"], dtype=np.float32)
            class_hits[names[int(np.argmax(tags.mean(axis=0)))]] += 1
            n_written += 1
            if split == "test" and clip_id in intervals:
                ap_scores[clip_id] = tags[:, idx].mean(axis=1)
                ap_labels[clip_id] = M.snippet_labels(int(row["T"]), 16.0,
                                                      intervals[clip_id])

    elapsed = time.perf_counter() - t_start
    stats = {
        "n_clips": n_written,
        "n_new_this_run": n_done,
        "wall_time_s": round(elapsed, 1),
        "s_per_clip": round(elapsed / max(n_done, 1), 3),
        "no_audio_or_missing_video": no_audio,
        "top_classes": class_hits.most_common(10),
    }
    if ap_scores:
        stats["incident_energy_ap_global"] = round(M.ap_global(ap_scores, ap_labels), 4)
        ap_per_clip, _per_clip = M.ap_per_video(ap_scores, ap_labels)
        stats["incident_energy_ap_per_clip"] = round(ap_per_clip, 4)
        stats["incident_energy_n_test_clips"] = len(ap_scores)
    (aed.AED_DIR / "aed_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
