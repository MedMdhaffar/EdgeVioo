#!/usr/bin/env python3
"""Local Streamlit interface for reviewing UCA interval candidates.

Run from the XDVioAlt directory with:
    .venv/bin/streamlit run scripts/uca_review_app.py

Reviewer decisions are saved back to the review CSV and preserved by the preparation script.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REVIEW_CSV = ROOT / "data/annotations/uca_intervals_review.csv"
LABELS = ["", "abuse", "fighting", "shooting", "riot", "car_accident", "explosion", "normal", "ignore"]


def load_rows() -> list[dict[str, str]]:
    if not REVIEW_CSV.is_file():
        st.error(f"Review table not found: {REVIEW_CSV}")
        st.stop()
    with REVIEW_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_rows(rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    fd, temporary = tempfile.mkstemp(prefix="uca_review_", suffix=".csv", dir=REVIEW_CSV.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, REVIEW_CSV)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


st.set_page_config(page_title="UCA interval review", layout="wide")
st.title("UCA interval review")
st.caption("Keyword matches are suggestions. Watch the interval and context, then choose a reviewed label. Nothing is pre-labeled.")
rows = load_rows()

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    split_filter = st.selectbox("Split", ["train", "val", "test", "all"], index=0)
with col2:
    label_filter = st.selectbox("Candidate category", ["abuse", "all six", "fighting", "shooting", "riot", "car_accident", "explosion"], index=0)
with col3:
    include_unmatched = st.checkbox("Include intervals with no keyword candidates", value=False)

eligible: list[int] = []
for index, row in enumerate(rows):
    if split_filter != "all" and row["split"] != split_filter:
        continue
    candidates = set(filter(None, row["candidate_labels"].split("|")))
    if not include_unmatched and not candidates:
        continue
    if label_filter == "abuse" and "abuse" not in candidates:
        continue
    if label_filter not in ("abuse", "all six") and label_filter not in candidates:
        continue
    eligible.append(index)

if not eligible:
    st.info("No intervals match these filters.")
    st.stop()

selection_key = "uca_review_selectbox"
if selection_key not in st.session_state or st.session_state[selection_key] not in eligible:
    st.session_state[selection_key] = eligible[0]

position = eligible.index(st.session_state[selection_key])
nav1, nav2, nav3 = st.columns([1, 4, 1])
with nav1:
    if st.button("← Previous", disabled=position == 0, use_container_width=True):
        st.session_state[selection_key] = eligible[position - 1]
        st.rerun()
with nav2:
    selected = st.selectbox(
        f"Interval {position + 1} of {len(eligible)}",
        eligible,
        index=position,
        format_func=lambda i: f"{rows[i]['video_name']} | {rows[i]['start_sec']}–{rows[i]['end_sec']} s | {rows[i]['candidate_labels'] or 'no candidate'}",
        key=selection_key,
    )
with nav3:
    if st.button("Next →", disabled=position == len(eligible) - 1, use_container_width=True):
        st.session_state[selection_key] = eligible[position + 1]
        st.rerun()

row = rows[selected]
left, right = st.columns([1.2, 1])
with left:
    st.subheader(row["video_name"])
    st.write(f"**UCA split:** {row['split']}  ·  **Candidate labels:** {row['candidate_labels'] or 'none'}")
    st.write(f"**Interval:** {row['start_sec']}–{row['end_sec']} s  ·  **Video duration:** {row['duration_sec']} s")
    st.markdown("**UCA description**")
    st.write(row["description"])
    st.caption("Choose `normal` only when you have verified the interval is a negative. Use `ignore` when uncertain or outside this project's labels.")
with right:
    video_path = ROOT / row["video_path"]
    if video_path.is_file():
        try:
            video_bytes = video_path.read_bytes()
            st.video(video_bytes, start_time=max(0, int(float(row["start_sec"]))),
                     end_time=max(1, int(float(row["end_sec"]) + 1)), muted=False)
        except OSError as exc:
            st.error(f"Could not read video: {exc}")
    else:
        st.error(f"Local video not found: {row['video_path']}")

current_label = row.get("review_label", "")
if current_label not in LABELS:
    current_label = ""
decision_col, notes_col = st.columns([1, 3])
with decision_col:
    reviewed_label = st.selectbox("Reviewed label", LABELS, index=LABELS.index(current_label), key=f"label_{selected}")
with notes_col:
    reviewed_notes = st.text_input("Review notes", value=row.get("review_notes", ""), key=f"notes_{selected}")

if st.button("Save review", type="primary"):
    row["review_label"] = reviewed_label
    row["review_notes"] = reviewed_notes
    save_rows(rows)
    st.success(f"Saved review for {row['video_name']} at {row['start_sec']}–{row['end_sec']} s.")

reviewed_train = sum(r["split"] == "train" and bool((r.get("review_label") or "").strip()) for r in rows)
st.caption(f"Reviewed training intervals: {reviewed_train}. CSV: {REVIEW_CSV.relative_to(ROOT)}")
