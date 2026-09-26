#!/usr/bin/env python3
"""Inspect and summarize outputs produced by run_five_crop_inference.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parent / "five_crop_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_RESULTS,
        help=f"results directory (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="anomaly threshold between 0 and 1 (default: 0.5)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of strongest snippets to print (default: 10)",
    )
    parser.add_argument(
        "--score",
        choices=("offline", "online"),
        default="offline",
        help="score used for ranking and thresholding (default: offline)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save offline/online curves as scores.png (requires matplotlib)",
    )
    return parser.parse_args()


def load_array(directory: Path, name: str) -> np.ndarray:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    values = np.load(path, allow_pickle=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return values


def load_csv(directory: Path) -> list[dict[str, str]]:
    path = directory / "scores.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def merged_intervals(
    rows: list[dict[str, str]], score_name: str, threshold: float
) -> list[tuple[float, float, float, int]]:
    """Merge adjacent snippets above threshold into continuous intervals."""
    intervals: list[tuple[float, float, float, int]] = []
    start = end = peak = None
    count = 0

    for row in rows:
        row_start = float(row["start_seconds"])
        row_end = float(row["end_seconds"])
        score = float(row[score_name])
        if score >= threshold:
            if start is None:
                start, end, peak, count = row_start, row_end, score, 1
            elif np.isclose(row_start, end):
                end = row_end
                peak = max(float(peak), score)
                count += 1
            else:
                intervals.append((float(start), float(end), float(peak), count))
                start, end, peak, count = row_start, row_end, score, 1
        elif start is not None:
            intervals.append((float(start), float(end), float(peak), count))
            start = end = peak = None
            count = 0

    if start is not None:
        intervals.append((float(start), float(end), float(peak), count))
    return intervals


def save_plot(
    directory: Path,
    rows: list[dict[str, str]],
    threshold: float,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to use --plot") from error

    times = [float(row["end_seconds"]) for row in rows]
    offline = [float(row["offline"]) for row in rows]
    online = [float(row["online"]) for row in rows]

    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(times, offline, label="Offline", linewidth=1.8)
    axis.plot(times, online, label="Online", linewidth=1.3, alpha=0.8)
    axis.axhline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:g}")
    axis.set(xlabel="Time (seconds)", ylabel="Violence score", ylim=(0, 1))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    destination = directory / "scores.png"
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def main() -> None:
    args = parse_args()
    directory = args.results_dir.resolve()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.top < 0:
        raise ValueError("--top cannot be negative")

    arrays = {
        **{f"rgb_{index}.npy": load_array(directory, f"rgb_{index}.npy") for index in range(5)},
        "audio.npy": load_array(directory, "audio.npy"),
        "timestamps_ms.npy": load_array(directory, "timestamps_ms.npy"),
        "video_model_input_5crop.npy": load_array(directory, "video_model_input_5crop.npy"),
        "offline_snippet_scores.npy": load_array(directory, "offline_snippet_scores.npy"),
        "online_snippet_scores.npy": load_array(directory, "online_snippet_scores.npy"),
        "offline_frame_scores.npy": load_array(directory, "offline_frame_scores.npy"),
        "online_frame_scores.npy": load_array(directory, "online_frame_scores.npy"),
    }
    rows = load_csv(directory)

    manifest_path = directory / "manifest.json"
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    temporal_length = arrays["audio.npy"].shape[0]
    expected = {
        **{f"rgb_{index}.npy": (temporal_length, 1024) for index in range(5)},
        "audio.npy": (temporal_length, 128),
        "timestamps_ms.npy": (temporal_length,),
        "video_model_input_5crop.npy": (5, temporal_length, 1152),
        "offline_snippet_scores.npy": (temporal_length,),
        "online_snippet_scores.npy": (temporal_length,),
        "offline_frame_scores.npy": (temporal_length * 16,),
        "online_frame_scores.npy": (temporal_length * 16,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {arrays[name].shape}")
    if len(rows) != temporal_length:
        raise ValueError(f"scores.csv: expected {temporal_length} rows, got {len(rows)}")

    print(f"Results directory: {directory}")
    print(f"Source video:      {manifest.get('source', 'unknown')}")
    print(f"Snippets:          {temporal_length}")
    print(f"Duration scored:   {temporal_length * 16 / 24:.3f} seconds")
    print("\nFiles:")
    for name, values in arrays.items():
        print(f"  {name:<32} shape={str(values.shape):<16} dtype={values.dtype}")

    ranked = sorted(rows, key=lambda row: float(row[args.score]), reverse=True)
    print(f"\nTop {min(args.top, len(ranked))} {args.score} snippets:")
    print("  snippet   start      end       score")
    for row in ranked[: args.top]:
        print(
            f"  {int(row['snippet']):>7}  "
            f"{float(row['start_seconds']):>7.3f}  "
            f"{float(row['end_seconds']):>7.3f}  "
            f"{float(row[args.score]):>10.6f}"
        )

    intervals = merged_intervals(rows, args.score, args.threshold)
    print(f"\nMerged {args.score} intervals at threshold >= {args.threshold:g}:")
    if not intervals:
        print("  None")
    else:
        print("  start      end       peak       snippets")
        for start, end, peak, count in intervals:
            print(f"  {start:>7.3f}  {end:>7.3f}  {peak:>10.6f}  {count:>8}")

    if args.plot:
        print(f"\nPlot saved to: {save_plot(directory, rows, args.threshold)}")


if __name__ == "__main__":
    main()
