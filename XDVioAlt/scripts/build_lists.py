#!/usr/bin/env python
"""Build train/val/test lists for XD-Violence (HF mirror) plus per-clip metadata.

Writes:
    data/lists/train.csv, val.csv, test.csv  -> clip_id, split, path, T, binary, labels
    data/annotations/test_intervals.csv      -> clip_id, start_frame, end_frame (tidy long form)

The validation split is carved out of the train split (clip-level labels only), so the test
frame-level annotations are never used for model selection.

Usage:
    .venv/bin/python scripts/build_lists.py --feature-set swin_rgb --val-size 400
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/raw/xd-violence/data")
OUT_LISTS = Path("data/lists")
OUT_ANN = Path("data/annotations")
# Feature folders per clip-index range (train split), as stored in the mirror.
TRAIN_CHUNKS = ("1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954")
LABEL_CODES = {
    "A": "normal",
    "B1": "fighting",
    "B2": "shooting",
    "B4": "riot",
    "B5": "abuse",
    "B6": "car_accident",
    "G": "explosion",
}


def source_of(clip_id: str) -> str:
    """Source video of a clip: everything before ``__#`` (a film name or a YouTube id).

    Clips sharing a source must never be split across train and validation (see ``make_val``),
    otherwise the model recognises the *film* instead of the *event*.
    """
    return clip_id.split("__#")[0] if "__#" in clip_id else clip_id


def parse_labels(clip_id: str) -> tuple[int, list[str]]:
    """Binary target + multi-label categories from the `_label_XXX` suffix ('0' = padding)."""
    if "_label_" not in clip_id:
        return 0, []
    codes = [c for c in clip_id.split("_label_")[-1].split("-") if c and c != "0"]
    cats = [LABEL_CODES[c] for c in codes if c in LABEL_CODES and c != "A"]
    return (1 if cats else 0), cats


def n_snippets(path: Path) -> int:
    """T of the feature tensor (memory-mapped, so this stays cheap for 4 750 files)."""
    import numpy as np

    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except Exception:  # noqa: BLE001 - one broken file must not abort the whole listing
        return 0


def parse_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    """`<id>.mp4 start end [start end ...]` (frames) -> {clip_id: [(start, end), ...]}."""
    ann: dict[str, list[tuple[int, int]]] = {}
    if not path.exists():
        return ann
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        ann[parts[0].removesuffix(".mp4")] = [
            (int(parts[i]), int(parts[i + 1])) for i in range(1, len(parts) - 1, 2)
        ]
    return ann


def _row(feature_path: Path, split: str) -> dict:
    binary, cats = parse_labels(feature_path.stem)
    return {
        "clip_id": feature_path.stem,
        "split": split,
        "path": str(feature_path),
        "T": n_snippets(feature_path),
        "binary": binary,
        "labels": "|".join(cats),
    }


def collect(feature_set: str) -> tuple[list[dict], list[dict]]:
    """One row per clip for the train chunks and for the test folder."""
    rows_train: list[dict] = []
    rows_test: list[dict] = []
    for chunk in TRAIN_CHUNKS:
        for f in sorted((ROOT / feature_set / chunk).glob("*.npy")):
            rows_train.append(_row(f, "train"))
    for f in sorted((ROOT / feature_set / "test_videos").glob("*.npy")):
        rows_test.append(_row(f, "test"))
    return rows_train, rows_test


def make_val(rows: list[dict], val_size: int, seed: int = 42,
             split_mode: str = "group") -> tuple[list[dict], list[dict]]:
    """Carve a validation split out of train.

    ``split_mode="group"`` (default) keeps **all clips of the same source together** (source =
    the part of the clip id before ``__#``: a film or a YouTube video). A random clip-level split
    leaks the source: the model then recognises the film instead of the event, and validation AP
    explodes (measured: 0.98 val vs 0.75-0.81 test with the random split). Groups are taken in a
    category-stratified way so the val set keeps roughly the train category mix.
    """
    if split_mode == "random":
        rng = random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[val_size:], shuffled[:val_size]

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[source_of(row["clip_id"])].append(row)

    group_category: dict[str, str] = {}
    for source, items in by_source.items():
        counts = Counter((r["labels"] or "normal").split("|")[0] for r in items)
        group_category[source] = counts.most_common(1)[0][0]

    by_category: dict[str, list[str]] = defaultdict(list)
    for source, category in group_category.items():
        by_category[category].append(source)

    rng = random.Random(seed)
    total = len(rows)
    val_sources: set[str] = set()
    for _category, sources in sorted(by_category.items()):
        rng.shuffle(sources)
        quota = val_size * sum(len(by_source[s]) for s in sources) / total
        taken = 0
        for source in sources:
            if taken >= quota:
                break
            val_sources.add(source)
            taken += len(by_source[source])

    val = [r for r in rows if source_of(r["clip_id"]) in val_sources]
    train = [r for r in rows if source_of(r["clip_id"]) not in val_sources]
    return train, val


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "split", "path", "T", "binary", "labels"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", default="swin_rgb",
                        help="feature folder under data/raw/xd-violence/data (swin_rgb, i3d_rgb)")
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", default="group", choices=["group", "random"],
                        help="group = source-disjoint validation split (honest model selection)")
    args = parser.parse_args()

    rows_train, rows_test = collect(args.feature_set)
    if not rows_train or not rows_test:
        print(f"[!] no features found for {args.feature_set} - run the download first")
        return 1
    train, val = make_val(rows_train, args.val_size, args.seed, args.split_mode)
    write_csv(train, OUT_LISTS / "train.csv")
    write_csv(val, OUT_LISTS / "val.csv")
    write_csv(rows_test, OUT_LISTS / "test.csv")

    split_info = {
        "feature_set": args.feature_set,
        "split_mode": args.split_mode,
        "val_size_requested": args.val_size,
        "seed": args.seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(rows_test),
        "n_sources_train": len({source_of(r["clip_id"]) for r in train}),
        "n_sources_val": len({source_of(r["clip_id"]) for r in val}),
        "source_overlap": len({source_of(r["clip_id"]) for r in train}
                              & {source_of(r["clip_id"]) for r in val}),
    }
    (OUT_LISTS / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print(f"[split] mode={args.split_mode} train={len(train)} val={len(val)} "
          f"sources train/val={split_info['n_sources_train']}/{split_info['n_sources_val']} "
          f"overlap={split_info['source_overlap']}")

    ann = parse_annotations(ROOT / "test_annotations.txt")
    OUT_ANN.mkdir(parents=True, exist_ok=True)
    with (OUT_ANN / "test_intervals.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip_id", "start", "end"])
        for clip_id, intervals in sorted(ann.items()):
            for start, end in intervals:
                writer.writerow([clip_id, start, end])

    for name, rows in (("train", train), ("val", val), ("test", rows_test)):
        cats = Counter((r["labels"] or "normal").split("|")[0] for r in rows)
        ts = sorted(r["T"] for r in rows if r["T"])
        median_t = ts[len(ts) // 2] if ts else 0
        print(f"{name:5s} clips={len(rows):5d} violent={sum(r['binary'] for r in rows):5d} "
              f"T(median)={median_t:4d} total_snippets={sum(ts):7d}")
        print(f"      first-label: {dict(cats.most_common())}")
    with_ann = sum(1 for r in rows_test if r["clip_id"] in ann)
    print(f"[info] test clips with frame-level intervals: {with_ann}/{len(rows_test)} "
          f"({sum(len(v) for v in ann.values())} intervals)")
    print(f"[ok] lists -> {OUT_LISTS}/  annotations -> {OUT_ANN / 'test_intervals.csv'}")
    print(f"[note] feature set = {args.feature_set}; snippet grid ~63 frames (~2.63 s at 24 fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

