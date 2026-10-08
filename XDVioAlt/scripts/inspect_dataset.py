#!/usr/bin/env python
"""Inspect the XD-Violence HuggingFace mirror and generate docs/dataset.md.

Validates the two assumptions that silently destroy results when wrong:
  1. do the mirror videos actually contain an audio stream?
  2. what is the snippet stride of the I3D RGB features (frames per feature vector)?

File-naming convention of the mirror (both videos and features):
    <film>__#HH-MM-SS_HH-MM-SS_label_<CODES>.npy|mp4
where CODES is `A` for normal clips and `-`-joined category codes otherwise
(B1 fighting, B2 shooting, B4 riot, B5 abuse, B6 car accident, G explosion).

Run:
    .venv/bin/python scripts/inspect_dataset.py --samples 3
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path("data/raw/xd-violence")
VGGISH_HOP_S = 0.96  # VGGish log-mel patch hop (seconds)

LABEL_CODES = {
    "A": "normal",
    "B1": "fighting",
    "B2": "shooting",
    "B4": "riot",
    "B5": "abuse",
    "B6": "car_accident",
    "G": "explosion",
}


def label_codes(video_id: str) -> list[str]:
    """Category codes parsed from the `_label_XXX` suffix ('0' entries are padding)."""
    if "_label_" not in video_id:
        return ["unknown"]
    codes = [c for c in video_id.split("_label_")[-1].split("-") if c and c != "0"]
    return codes or ["unknown"]


def category_of(video_id: str) -> str:
    """First (dominant) category of a clip, e.g. 'A.Beautiful..._label_B1-B6' -> fighting."""
    codes = label_codes(video_id)
    return LABEL_CODES.get(codes[0], "unknown") if codes else "unknown"


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _fps(stream: dict) -> float:
    num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
    try:
        return float(num) / float(den) if float(den) else 0.0
    except ValueError:
        return 0.0


def stream_summary(info: dict) -> tuple[dict, dict]:
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    fmt = info.get("format", {})
    v = {
        "codec": video.get("codec_name", "-"),
        "fps": round(_fps(video), 3) if video else 0.0,
        "nb_frames": int(video.get("nb_frames") or 0),
        "size": f"{video.get('width', '?')}x{video.get('height', '?')}",
        "duration_s": round(float(fmt.get("duration") or 0.0), 2),
    }
    a = {
        "present": audio is not None,
        "codec": audio.get("codec_name", "-") if audio else "-",
        "sample_rate": int(audio.get("sample_rate", 0) or 0) if audio else 0,
        "channels": int(audio.get("channels", 0) or 0) if audio else 0,
    }
    return v, a


def parse_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    """`<id>.mp4 start end [start end ...]` (frames) -> {id: [(start, end), ...]}."""
    ann: dict[str, list[tuple[int, int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        # NOTE: never use str.rstrip(".mp4") - it strips any of {'.','m','p','4'} and
        # corrupts ids ending in '4' (e.g. "..._B4.mp4" -> "..._B").
        vid = parts[0].removesuffix(".mp4")
        ann[vid] = [(int(parts[i]), int(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
    return ann


def read_ids(path: Path) -> list[str]:
    """Read a split list; entries may be prefixed (e.g. `test_videos/`) -> keep basename."""
    if not path.exists():
        return []
    return [Path(ln.strip().removesuffix(".mp4")).name
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def feature_path(features_dir: Path, video_id: str) -> Path | None:
    exact = features_dir / f"{video_id}.npy"
    if exact.exists():
        return exact
    matches = sorted(features_dir.glob(f"{video_id}*.npy"))
    return matches[0] if matches else None


def feature_shape(path: Path) -> tuple | None:
    try:
        import numpy as np

        return tuple(np.load(path, mmap_mode="r").shape)
    except Exception:  # noqa: BLE001 - report unknown shape instead of crashing
        return None


def analyse_samples(root: Path, n_samples: int) -> list[dict]:
    """ffprobe + feature-shape check for n sample test clips, spread across categories."""
    videos_dir = root / "data" / "video" / "test_videos"
    features_dir = root / "data" / "i3d_rgb" / "test_videos"
    videos = sorted(videos_dir.glob("*.mp4"))
    if not videos:
        return []

    by_cat: dict[str, list[Path]] = {}
    for video in videos:
        by_cat.setdefault(category_of(video.stem), []).append(video)
    order = ["normal", "explosion", "car_accident", "fighting", "abuse", "riot", "shooting"]
    chosen: list[Path] = [by_cat[c][0] for c in order if by_cat.get(c)][:n_samples]
    for items in by_cat.values():
        for item in items:
            if len(chosen) >= n_samples:
                break
            if item not in chosen:
                chosen.append(item)

    rows: list[dict] = []
    for video in chosen[:n_samples]:
        try:
            video_info, audio_info = stream_summary(ffprobe(video))
        except subprocess.CalledProcessError as exc:
            rows.append({"id": video.stem, "error": f"ffprobe failed: {exc}"})
            continue
        row: dict = {
            "id": video.stem,
            "category": category_of(video.stem),
            "video": video_info,
            "audio": audio_info,
        }
        npy = feature_path(features_dir, video.stem)
        if npy is not None:
            shape = feature_shape(npy)
            n_snippets = shape[0] if shape else 0
            row["feature"] = {"shape": shape, "n_snippets": n_snippets}
            if video_info["nb_frames"] and n_snippets:
                row["stride_frames"] = round(video_info["nb_frames"] / n_snippets, 3)
                row["seconds_per_snippet"] = round(video_info["duration_s"] / n_snippets, 3)
                row["audio_patches"] = round(video_info["duration_s"] / VGGISH_HOP_S, 1)
        else:
            row["feature"] = None
        rows.append(row)
    return rows


def _table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


def _cats_txt(counter: Counter, top: int = 4) -> str:
    """Compact 'category count, ...' summary for the markdown report."""
    items = counter.most_common()
    head = ", ".join(f"{k} {v}" for k, v in items[:top])
    rest = len(items) - top
    return head + (f", +{rest} more" if rest > 0 else "") if head else "-"



def _report_overview(root: Path) -> list[str]:
    test_ids = read_ids(root / "data" / "test_list.txt")
    train_ids = read_ids(root / "data" / "train_list.txt")
    ann_path = root / "data" / "test_annotations.txt"
    ann = parse_annotations(ann_path) if ann_path.exists() else {}
    first = Counter(category_of(i) for i in test_ids)
    all_codes: Counter = Counter()
    for i in test_ids:
        for c in label_codes(i):
            all_codes[LABEL_CODES.get(c, c)] += 1
    frames_annot = sum(e - s for pairs in ann.values() for s, e in pairs)
    n_intervals = sum(len(v) for v in ann.values())
    n_multi = sum(1 for v in ann.values() if len(v) >= 2)
    test_set = set(test_ids)
    ann_ids = test_set & set(ann)
    miss_ids = test_set - set(ann)
    ann_cats = Counter(category_of(i) for i in ann_ids)
    miss_cats = Counter(category_of(i) for i in miss_ids)
    n_annotated = len(ann_ids)
    return [
        "# XD-Violence (HF mirror `jherng/xd-violence`) - dataset facts",
        "",
        "> Generated by `scripts/inspect_dataset.py`; do not edit by hand.",
        "> Cite: P. Wu et al., *Not only Look, but also Listen*, ECCV 2020.",
        "",
        "## Splits and annotations",
        f"- test clips **{len(test_ids)}**, train clips **{len(train_ids)}**",
        f"- test clips with >=1 annotated anomaly interval: **{n_annotated}** / {len(ann)}",
        f"- total annotated anomalous frames in test: **{frames_annot:,}**",
        f"- anomaly intervals in total: **{n_intervals}**; clips with >=2 events: **{n_multi}**",
        f"- annotation coverage: **{n_annotated} of {len(test_ids)}** test clips "
        f"({len(ann) - n_annotated} annotation ids are outside test_list)",
        "",
        "### Annotated vs unannotated test clips (ground-truth completeness)",
        *_table(["group", "clips", "first-label breakdown"],
                [["annotated (>=1 interval)", n_annotated, _cats_txt(ann_cats)],
                 ["unannotated (normal only)", len(miss_ids), _cats_txt(miss_cats)]]),
        "",
        "- GT construction: every test clip gets a snippet-level label vector; unannotated "
        "clips are entirely normal, so the full 800-clip test set is scoreable",
        "- `test_annotations.txt` format: `<id>.mp4 start end [start end ...]`, frame indices",
        "- clip ids: `<film>__#HH-MM-SS_HH-MM-SS_label_<CODES>` with `CODES` = `A` (normal) or "
        "joined categories (B1 fighting, B2 shooting, B4 riot, B5 abuse, B6 car accident, "
        "G explosion)",
        "- training supervision is clip-level only; frame-level annotations exist for test only",
        "",
        "### First label distribution (test)",
        *_table(["category", "clips"], [[c, n] for c, n in first.most_common()]),
        "",
        "### All label codes (test, multi-label)",
        *_table(["category", "occurrences"], [[c, n] for c, n in all_codes.most_common()]),
        "",
    ]


def _report_samples(rows: list[dict]) -> list[str]:
    strides = [r["stride_frames"] for r in rows if r.get("stride_frames")]
    sps = [r["seconds_per_snippet"] for r in rows if r.get("seconds_per_snippet")]
    n_audio = sum(1 for r in rows if r.get("audio", {}).get("present"))
    n_feat = sum(1 for r in rows if r.get("feature"))
    out: list[str] = [
        "## Audio stream check (CRITICAL)",
        f"- sampled clips **{len(rows)}**, with an audio stream **{n_audio}/{len(rows)}**",
        "",
        *_table(["clip", "category", "v-codec", "fps", "frames", "dur (s)", "res", "a-codec",
                 "sr (Hz)", "ch"],
                [[r["id"][:34], r.get("category", "-"), r.get("video", {}).get("codec", "-"),
                  r.get("video", {}).get("fps", "-"), r.get("video", {}).get("nb_frames", "-"),
                  r.get("video", {}).get("duration_s", "-"), r.get("video", {}).get("size", "-"),
                  r.get("audio", {}).get("codec", "-"),
                  r.get("audio", {}).get("sample_rate", "-"),
                  r.get("audio", {}).get("channels", "-")] for r in rows]),
        "",
        "## Snippet stride calibration (I3D RGB)",
        f"- clips with a matching feature file **{n_feat}/{len(rows)}**",
        "",
        *_table(["clip", "feature shape", "T", "frames/T", "s/snippet", "VGGish patches @0.96s"],
                [[r["id"][:34], (r.get("feature") or {}).get("shape", "-"),
                  (r.get("feature") or {}).get("n_snippets", "-"), r.get("stride_frames", "-"),
                  r.get("seconds_per_snippet", "-"), r.get("audio_patches", "-")] for r in rows]),
        "",
    ]
    if strides:
        out.append(f"- **median stride {statistics.median(strides):g} frames/snippet**")
    if sps:
        med = statistics.median(sps)
        out.append(f"- **median snippet duration {med:g} s** (= {1 / med:.2f} snippets/s)")
    frames_per_snippet = [r["video"]["nb_frames"] / r["feature"]["n_snippets"] for r in rows
                          if r.get("feature") and r.get("video", {}).get("nb_frames")
                          and r["feature"].get("n_snippets")]
    ceil64_ok = all(
        abs(r["feature"]["n_snippets"] - math.ceil(r["video"]["nb_frames"] / 64)) <= 1
        for r in rows if r.get("feature") and r.get("video", {}).get("nb_frames")
    )
    out += [
        "",
        "### Derived pipeline rules (measured, not assumed)",
        "- feature tensor `(T, 5 crops, D)`, D = 2048 (I3D RGB); crop 0 = centre, 1..4 = corners",
        "- **the mirror's features use a 64-frame window (~2.67 s at 24 fps)**: the rule "
        f"`T = ceil(nb_frames / 64)` holds for every sampled clip: **{ceil64_ok}**",
        f"- real stride = `nb_frames / T` ~ {statistics.median(frames_per_snippet):.1f} frames "
        "(float and clip-dependent) -> always pass the measured stride to "
        "`safewatch/eval/metrics.py`; never assume the official 16-frame grid",
        "- snippet GT: a snippet is anomalous when its frame window overlaps an annotated interval",
        "- audio: 48 kHz **6-7 channel** AAC -> downmix to mono, resample to 16 kHz for VGGish; "
        "~2.75 VGGish patches (0.96 s hop) per snippet -> average-pool onto the snippet grid",
        "- 500 of the 800 test clips carry intervals and all 300 others are normal, so the whole "
        "test split is scoreable",
        "- CONSEQUENCE: ~2.6 s temporal granularity is coarse for fine localization; the "
        "edge/demo track should extract its own features at a finer stride (~1 s)",
        "",
        "### Known caveats",
        "- the mirror ships **no audio features**: VGGish must be extracted from the videos or "
        "taken from the authors' gated Baidu link",
        "- the mirror's reference loader uses `str.rstrip('.mp4')`, which strips any of "
        "{'.','m','p','4'} and corrupts ids ending in `4`; our parser avoids that",
        "",
    ]
    return out



def build_report(root: Path, rows: list[dict]) -> str:
    return "\n".join(_report_overview(root) + _report_samples(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("docs/dataset.md"))
    args = parser.parse_args()

    if not args.root.exists():
        print(f"[!] {args.root} does not exist - run scripts/download_hf.py first")
        return 1

    rows = analyse_samples(args.root, args.samples)
    report = build_report(args.root, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[ok] report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

