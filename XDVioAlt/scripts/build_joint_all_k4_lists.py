#!/usr/bin/env python
"""Create the unified joint k4_mstd (2048-d) training lists:
XD-Violence + RWF-2000 + SCFD + SmartDashcam + AccidentVideos.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xd-lists", default="data/lists_binary_minirgb_sourceholdout_k4_mstd")
    parser.add_argument("--rwf-lists", default="data/lists_rwf2000_binary_k4_mstd")
    parser.add_argument("--scfd-lists", default="data/lists_scfd_binary_k4_mstd")
    parser.add_argument("--dashcam-lists", default="data/lists_dashcam_binary_k4_mstd")
    parser.add_argument("--caracc-lists", default="data/lists_caracc_binary_k4_mstd")
    parser.add_argument("--out", default="data/lists_joint_all_k4_mstd")
    args = parser.parse_args()

    out = ROOT / args.out

    xd_train = read_csv(ROOT / args.xd_lists / "train.csv")
    xd_val = read_csv(ROOT / args.xd_lists / "val.csv")
    xd_test = read_csv(ROOT / args.xd_lists / "test.csv")
    for r in xd_train + xd_val + xd_test:
        r.setdefault("dataset", "XD-Violence")

    rwf_train = read_csv(ROOT / args.rwf_lists / "train.csv")
    rwf_val = read_csv(ROOT / args.rwf_lists / "val.csv")

    scfd_train = read_csv(ROOT / args.scfd_lists / "train.csv")
    scfd_val = read_csv(ROOT / args.scfd_lists / "val.csv")

    dash_train = read_csv(ROOT / args.dashcam_lists / "train.csv")
    dash_val = read_csv(ROOT / args.dashcam_lists / "val.csv")

    caracc_train = read_csv(ROOT / args.caracc_lists / "train.csv")
    caracc_val = read_csv(ROOT / args.caracc_lists / "val.csv")

    joint_train = xd_train + rwf_train + scfd_train + dash_train + caracc_train
    joint_val = xd_val + rwf_val + scfd_val + dash_val + caracc_val
    joint_test = xd_test

    write_csv(out / "train.csv", joint_train)
    write_csv(out / "val.csv", joint_val)
    write_csv(out / "test.csv", joint_test)

    summary = {
        "dataset": "Joint XD-Violence + RWF-2000 + SCFD + SmartDashcam + AccidentVideos (k4_mstd 2048-d)",
        "splits": {
            "train": {
                "total": len(joint_train),
                "xd": len(xd_train),
                "rwf": len(rwf_train),
                "scfd": len(scfd_train),
                "dashcam": len(dash_train),
                "caracc": len(caracc_train),
            },
            "val": {
                "total": len(joint_val),
                "xd": len(xd_val),
                "rwf": len(rwf_val),
                "scfd": len(scfd_val),
                "dashcam": len(dash_val),
                "caracc": len(caracc_val),
            },
            "test": {"total": len(joint_test)},
        },
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
