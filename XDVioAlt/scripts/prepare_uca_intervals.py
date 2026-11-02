#!/usr/bin/env python3
"""Download UCA annotations and make a reviewable interval table for local UCF videos.

UCA descriptions are event captions, not this project's six-class violence labels. The keyword
mapping below therefore emits *candidates only*. Nothing written by this script is training-ready
until a reviewer sets ``review_label`` to a class, ``normal`` (verified negative), or ``ignore``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = (
    "https://raw.githubusercontent.com/Xuange923/Surveillance-Video-Understanding/"
    "main/UCF%20Annotation/json/"
)
FILES = ("UCFCrime_Train.json", "UCFCrime_Val.json", "UCFCrime_Test.json")

# Conservative phrase cues for candidate review only; not an automatic ground-truth map.
PATTERNS: dict[str, tuple[str, ...]] = {
    "fighting": (r"fight(?:s|ing)?", r"punch(?:es|ed|ing)?", r"kick(?:s|ed|ing)?", r"brawl", r"beating"),
    "shooting": (r"shoot(?:s|ing)?", r"gunfire", r"firearm", r"fires? (?:a )?(?:gun|shot)", r"shot at"),
    "riot": (r"riot(?:s|ing)?", r"mob attack", r"crowd attack", r"violent crowd"),
    "abuse": (r"abuse", r"assault(?:s|ed|ing)?", r"attacks?", r"hits? (?:the|a|an|him|her|them)",
              r"slaps?", r"strangles?", r"chokes?", r"stabs?", r"beating"),
    "car_accident": (r"collision", r"crash(?:es|ed|ing)?", r"car accident", r"vehicle accident", r"runs? over"),
    "explosion": (r"explod(?:e|es|ed|ing)", r"explosion", r"blast(?:s|ed)?", r"bomb(?:s|ed)?"),
}
COMPILED = {name: tuple(re.compile(p, re.I) for p in patterns) for name, patterns in PATTERNS.items()}
FIELDS = ["video_name", "split", "video_path", "start_sec", "end_sec", "duration_sec",
          "description", "candidate_labels", "review_label", "review_notes", "source_file"]


def fetch_annotations(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in FILES:
        destination = directory / filename
        if not destination.is_file():
            request = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": "XDVioAlt-UCA-preparer/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=45) as response, destination.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                destination.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download {filename} from the official UCA GitHub repository: {exc}. "
                    f"Network may be unavailable; download it manually into {directory} and rerun."
                ) from exc
        paths.append(destination)
    return paths


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def candidate_labels(description: str) -> list[str]:
    return [label for label, patterns in COMPILED.items() if any(pattern.search(description) for pattern in patterns)]


def review_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row.get("video_name", ""), row.get("start_sec", ""), row.get("end_sec", ""), row.get("description", ""))


def find_video(video_root: Path, video_name: str) -> Path | None:
    # UCA keys may include a subdirectory; match the basename against local files.
    basename = Path(video_name.replace("\\", "/")).name
    if not Path(basename).suffix:
        basename += ".mp4"
    direct = video_root / basename
    if direct.is_file():
        return direct
    matches = list(video_root.rglob(basename))
    return matches[0] if len(matches) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/raw/ucf_crime/uca/json")
    parser.add_argument("--videos", default="data/raw/ucf_crime/Videos")
    parser.add_argument("--output", default="data/annotations/uca_intervals_review.csv")
    args = parser.parse_args()
    annotations_dir, video_root = ROOT / args.annotations, ROOT / args.videos
    output = ROOT / args.output
    prior_reviews: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    if output.is_file():
        with output.open(newline="", encoding="utf-8") as fh:
            for old_row in csv.DictReader(fh):
                label, notes = old_row.get("review_label", ""), old_row.get("review_notes", "")
                if label or notes:
                    prior_reviews[review_key(old_row)] = (label, notes)
    files = fetch_annotations(annotations_dir)

    rows: list[dict[str, str]] = []
    stats: Counter[str] = Counter()
    for annotation_file in files:
        split = annotation_file.stem.removeprefix("UCFCrime_").lower()
        for video_name, record in read_json(annotation_file).items():
            stats["annotated_videos"] += 1
            video_path = find_video(video_root, video_name)
            if video_path is None:
                stats["no_local_video"] += 1
                continue
            timestamps = record.get("timestamps", [])
            sentences = record.get("sentences", [])
            if len(timestamps) != len(sentences):
                stats["bad_annotation_records"] += 1
                continue
            duration = float(record.get("duration", 0.0) or 0.0)
            for interval, sentence in zip(timestamps, sentences):
                if len(interval) != 2:
                    stats["bad_intervals"] += 1
                    continue
                start, end = map(float, interval)
                text = str(sentence).strip()
                if start < 0 or end <= start or (duration and end > duration + 0.25):
                    stats["bad_intervals"] += 1
                    continue
                labels = candidate_labels(text)
                normalized_video_name = Path(video_name.replace("\\", "/")).name
                if not Path(normalized_video_name).suffix:
                    normalized_video_name += ".mp4"
                output_row = {
                    "video_name": normalized_video_name,
                    "split": split,
                    "video_path": str(video_path.relative_to(ROOT)),
                    "start_sec": f"{start:.3f}",
                    "end_sec": f"{end:.3f}",
                    "duration_sec": f"{duration:.3f}",
                    "description": text,
                    "candidate_labels": "|".join(labels),
                    "review_label": "",  # Reviewer must set a valid class or 'ignore'.
                    "review_notes": "",
                    "source_file": annotation_file.name,
                }
                previous_label, previous_notes = prior_reviews.get(review_key(output_row), ("", ""))
                output_row["review_label"] = previous_label
                output_row["review_notes"] = previous_notes
                rows.append(output_row)
                stats["matched_intervals"] += 1
                if labels:
                    stats["candidate_intervals"] += 1
                else:
                    stats["no_keyword_candidate"] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "annotation_source": "Official UCA UCF-Crime annotation repository",
        "video_root": str(video_root.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "counts": dict(stats),
        "candidate_labels": list(PATTERNS),
        "label_policy": "Keyword matches are review candidates only. Blank review_label means excluded from supervised training. After viewing, set review_label to one of the six labels, 'normal' for a verified negative, or 'ignore'. Unannotated time is unknown, never an automatic negative.",
        "split_policy": "UCA train/val/test split is retained; use only train records for model fitting. Do not use UCA val/test as training. Preserve video-level split boundaries.",
        "limitations": [
            "UCA captions describe events and do not directly encode this project's six-label ontology.",
            "This output does not manufacture negative intervals or claim all local Abuse-category videos are project-defined abuse.",
            "The original UCF-Crime videos are separate from the UCA annotation repository; only locally available videos are matched.",
        ],
        "preserved_existing_reviews": len(prior_reviews),
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
