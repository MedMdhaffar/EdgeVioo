#!/usr/bin/env python
"""P6 - SafeWatch supervision console (Streamlit).

Usage:
    .venv/bin/python -m streamlit run app.py

What it is: the operator side of the pipeline. Per clip it shows the score curve on a **seconds**
axis (whatever grid the run uses), the predicted incident segments, and - when an incident sheet
exists - the model's category / confidence / modality evidence and its caveats. It then records the
operator's verdict in an **append-only** JSONL log, so a rejected alert stays visible instead of
vanishing.

Three deliberate design rules:

1. **The app is a renderer.** All state and every computation live in :mod:`safewatch.ui.state` (no
   Streamlit, unit-tested); figures live in :mod:`safewatch.ui.figures`. Nothing here re-implements
   the scoring, the grid conversion or the alert logic - the pipeline's artefacts are the contract.
2. **It never invents an alert.** If no incident sheet exists for a clip, the console says so and
   gives the exact ``scripts/incident_sheet.py`` command rather than a half-empty card.
3. **Ground truth is labelled as such.** On the test split the annotations are the evaluation
   reference, not a model output; the console marks them and never decides anything from them.

Generating the sheets to display (once per clip you want to review):
    .venv/bin/python scripts/incident_sheet.py --ckpt runs/<run>/ckpt_best.pt \\
        --clip <clip_id> --media
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importable under `streamlit run` too

from safewatch.ui import figures as F
from safewatch.ui import state as S

# Auto-reload all safewatch submodules so hot-reloading never has stale imports
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("safewatch.") and mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

DEFAULT_SHEETS = "sheets"
DEFAULT_DECISIONS = "sheets/decisions.jsonl"
DEFAULT_INTERVALS = "data/annotations/test_intervals.csv"


@st.cache_data(show_spinner=False)
def cached_runs(runs_dir: str, sheets_dir: str, champions_only: bool = False):
    return S.list_runs(runs_dir, alerts_dir=sheets_dir, champions_only=champions_only)


@st.cache_data(show_spinner="Loading predictions…")
def cached_predictions(run_path: str):
    scores, gt = S.load_predictions(run_path)
    if not scores:
        return {}, {}
    return scores, gt


@st.cache_data(show_spinner=False)
def cached_intervals(path: str):
    return S.load_intervals(path)


def sidebar() -> dict:
    """Run selection and display knobs - everything the main pane needs."""
    st.sidebar.title("SafeWatch")
    st.sidebar.caption("Supervision Console & Edge Demo — P6")

    mode = st.sidebar.radio(
        "Mode", ["Run Review", "Live Demo / Import Video"],
        format_func=lambda m: (
            "📁 Run Review" if m == "Run Review" else "🎥 Live Demo / Import"
        ))

    champions_only = st.sidebar.toggle(
        "Recommended Models Only", value=True,
        help="Shows the 3 top-performing champion models (Cross-Attention & Edge CPU)")

    runs_dir = st.sidebar.text_input("Runs Directory", "runs")
    sheets_dir = st.sidebar.text_input("Incident Sheets Directory", DEFAULT_SHEETS)
    decisions_path = st.sidebar.text_input("Decision Audit Log", DEFAULT_DECISIONS)

    runs = cached_runs(runs_dir, sheets_dir, champions_only=champions_only)
    chosen_run = None
    if runs:
        names = [run.name for run in runs]
        default = next((i for i, n in enumerate(names) if "p3full_late_clipnorm" in n), 0)
        chosen = st.sidebar.selectbox(
            "Run / Model", names, index=default,
            format_func=lambda n: f"{S.CHAMPION_RUNS.get(n, n)} [{n}]")
        chosen_run = runs[names.index(chosen)]
        ap_label = "n/a" if chosen_run.ap_global is None else f"{chosen_run.ap_global:.4f}"
        st.sidebar.metric("Global AP", ap_label)
        st.sidebar.caption(
            f"{chosen_run.n_clips} clips · grid {chosen_run.stride_frames:g} frames/snippet"
            f"{' · ' + chosen_run.fusion_mode if chosen_run.fusion_mode else ''}")
    elif mode == "Run Review":
        st.sidebar.error(f"No scored runs found in `{runs_dir}` (needs a `predictions.npz`).")
        st.stop()

    st.sidebar.divider()
    st.sidebar.subheader("Alert Rules")
    threshold = st.sidebar.slider("Probability Threshold", 0.05, 0.95, 0.5, 0.05)
    min_seconds = st.sidebar.number_input("Minimum Duration (s)", 0.5, 60.0,
                                          float(S.DEFAULT_MIN_SECONDS), 0.5)
    max_gap_seconds = st.sidebar.number_input("Gap Fusion (s)", 0.0, 30.0,
                                              float(S.DEFAULT_MAX_GAP_SECONDS), 0.5)
    adaptive = st.sidebar.toggle(
        "Category-Adaptive Segmentation", value=True,
        help="Adjusts minimum duration per category (e.g. 1.5s collision, 4s riot)")
    live_mode = mode == "Live Demo / Import Video"
    causal_label = ("Causal Prefix Scoring (Offline Simulation)" if live_mode
                    else "Causal EMA Smoothing (Display Only)")
    causal_help = (
        "Re-runs the model on prefixes of 8 snippets. Each emitted score uses only past/current "
        "snippets. The full video is still decoded and features are extracted before scoring; "
        "this is not live frame-by-frame inference."
        if live_mode else
        "Smooths cached full-clip predictions using only previous score values. This cannot make "
        "the original model inference causal.")
    causal = st.sidebar.toggle(causal_label, value=False, help=causal_help)
    if causal:
        if live_mode:
            st.sidebar.warning(
                "Prefix scoring repeats the model forward pass roughly T/8 times. "
                "Feature extraction still runs on the complete upload first.")
        else:
            st.sidebar.warning(
                "These cached model scores used the full clip. Only the displayed curve is "
                "smoothed causally.")

    order = "score_desc"
    show_gt = True
    if mode == "Run Review":
        order = st.sidebar.radio("Review Order", ["score_desc", "score_asc", "clip_id"],
                                 format_func=lambda key: {"score_desc": "Descending score",
                                                          "score_asc": "Ascending score",
                                                          "clip_id": "Clip ID"}[key])
        show_gt = st.sidebar.toggle(
            "Show Ground Truth", value=True,
            help="Evaluation reference from the test split, not a model output.")

    return {"mode": mode, "run": chosen_run, "threshold": threshold, "min_seconds": min_seconds,
            "max_gap_seconds": max_gap_seconds, "order": order, "show_gt": show_gt,
            "adaptive": adaptive, "causal": causal, "champions_only": champions_only,
            "sheets_dir": sheets_dir, "decisions_path": decisions_path,
            "runs_dir": runs_dir, "intervals_path": DEFAULT_INTERVALS}


def render_alert(alert: dict) -> None:
    """The incident card, exactly as ``scripts/incident_sheet.py`` wrote it (no reinvention)."""
    category = alert.get("category") or {}
    confidence = alert.get("confidence") or {}
    modality = alert.get("modality_contribution") or {}

    top = st.columns(3)
    top[0].metric("Category", category.get("label") or "n/a",
                  None if category.get("probability") is None
                  else f"p={category['probability']:.2f}")
    alert_probability = confidence.get("alert_probability", confidence.get("raw_alert_probability"))
    top[1].metric("Alert Probability (top-k)",
                  "n/a" if alert_probability is None else f"{alert_probability:.3f}",
                  "calibrated" if confidence.get("calibrated") else "raw")
    contribution = modality.get("mean_probability") or {}
    top[2].metric("Mean Contribution",
                  " / ".join(f"{k} {v:.2f}" for k, v in contribution.items()) or "n/a")

    cat_fig = F.category_figure(category)
    if cat_fig is not None:
        st.plotly_chart(cat_fig, width="stretch")

    if alert.get("segments"):
        st.dataframe(alert["segments"], width="stretch", hide_index=True)
    if alert.get("text"):
        st.code(alert["text"], language=None)

    with st.expander("Evidence (Attention, Audio phrases, Reliability)"):
        st.json({"visual_evidence": alert.get("visual_evidence"),
                 "audio_evidence": alert.get("audio_evidence"),
                 "modality_contribution": modality}, expanded=False)
    with st.expander("Model Caveats", expanded=False):
        for caveat in alert.get("caveats", []):
            st.warning(caveat, icon="⚠️")

    media = alert.get("media") or {}
    if media.get("spectrogram") or media.get("key_frames"):
        with st.expander("Media", expanded=False):
            if media.get("spectrogram") and Path(media["spectrogram"]).exists():
                st.image(media["spectrogram"], caption="Spectrogram of peak incident segment")
            zones = {z.get("second"): z for z in (media.get("motion") or [])}
            for frame in (media.get("key_frames") or [])[:5]:
                frame_path = frame.get("path") if isinstance(frame, dict) else frame
                if not (frame_path and Path(frame_path).exists()):
                    continue
                overlay = zones.get(frame.get("second"))
                overlay_path = overlay.get("overlay") if overlay else None
                if overlay_path and Path(overlay_path).exists():
                    boxes = ", ".join(f"{b['x']:.2f},{b['y']:.2f} "
                                      f"({b['w']:.2f}x{b['h']:.2f}, motion {b['motion']})"
                                      for b in overlay["boxes"])
                    st.image(overlay_path, width=260,
                             caption=f"Motion zones: {boxes}" if boxes else None)
                else:
                    st.image(frame_path, width=260)


def render_operator(clip_id: str, run_name: str, timeline: dict, decisions_path: str) -> None:
    """Confirm / reject / unsure - written to the append-only log, never overwritten."""
    st.subheader("Operator Decision")
    note = st.text_input("Note (optional)", key=f"note::{clip_id}",
                         placeholder="e.g. 'Real incident, but outside main camera frame'")
    operator = st.text_input("Operator", value="operator", key="operator_name")

    columns = st.columns(3)
    labels = {"confirmed": "✅ Confirm", "rejected": "⛔ Reject", "unsure": "❔ Unsure"}
    for column, verdict in zip(columns, S.DECISIONS, strict=True):
        if column.button(labels[verdict], key=f"btn::{verdict}::{clip_id}",
                         width="stretch"):
            S.record_decision(decisions_path, S.Decision(
                clip_id=clip_id, decision=verdict, run=run_name, operator=operator, note=note,
                clip_score=round(timeline["clip_score"], 4) if timeline else None,
                n_segments=len(timeline["segments"]) if timeline else 0))
            st.success(f"Decision '{verdict}' recorded for {clip_id}.")
            st.rerun()

    latest = S.decided_clips(decisions_path)
    if clip_id in latest:
        st.info(f"Latest verdict recorded for this clip: **{latest[clip_id]}** "
                "(the log is append-only: full audit history is preserved).")


def render_decisions(decisions_path: str) -> None:
    """Latest verdict per clip + the raw log as a download."""
    decisions = S.load_decisions(decisions_path)
    st.subheader(f"Decision Log ({len(decisions)} record(s))")
    if not decisions:
        st.caption("No decisions recorded yet.")
        return
    latest: dict[str, dict] = {}
    for decision in decisions:
        latest[decision.clip_id] = {"clip_id": decision.clip_id, "decision": decision.decision,
                                    "operator": decision.operator, "note": decision.note,
                                    "clip_score": decision.clip_score, "run": decision.run,
                                    "recorded_at": decision.recorded_at}
    st.dataframe(list(latest.values()), width="stretch", hide_index=True)
    path = Path(decisions_path)
    if path.exists():
        st.download_button("Download Decision Log (JSONL)", path.read_text(encoding="utf-8"),
                            file_name=path.name, mime="application/json")


def render_live_demo(options: dict) -> None:
    """Interactive video upload & real-time on-the-fly inference mode."""
    st.title("🎥 Live Demo — End-to-End Inference & Explainability")
    st.caption(
        "Upload a raw video or pick a test extract to run audio-visual extraction "
        "and incident detection on-the-fly.")

    checkpoints = S.list_checkpoints(
        options["runs_dir"], champions_only=options.get("champions_only", True)
    )
    if not checkpoints:
        st.error(f"No checkpoint (.pt) found in `{options['runs_dir']}`.")
        return

    ckpt_names = [f"{p.parent.name}/{p.name}" for p in checkpoints]
    default_ckpt_idx = next(
        (i for i, n in enumerate(ckpt_names) if "p3edge_i3dmel" in n and "best" in n),
        next((i for i, n in enumerate(ckpt_names)
              if "p3full_late_clipnorm" in n and "best" in n), 0))
    def _fmt_ckpt(name: str) -> str:
        run = name.split("/")[0]
        return f"{S.CHAMPION_RUNS.get(run, run)} [{run}]"

    selected_ckpt = st.selectbox(
        "Model Checkpoint", ckpt_names, index=default_ckpt_idx,
        format_func=_fmt_ckpt,
    )
    ckpt_path = checkpoints[ckpt_names.index(selected_ckpt)]
    supports_live, live_problems = S.live_feature_support(
        S.read_json(ckpt_path.parent / "config.json", {}) or {})
    if supports_live:
        st.caption("✅ This checkpoint was trained on features the live extractor can reproduce "
                   "(I3D visual + log-mel audio).")
    else:
        st.warning(
            "⚠️ This checkpoint was trained on a **different feature space** than the live "
            "extractor produces, so uploaded videos will score **near zero even on obvious "
            "incidents** (the model never saw these inputs). " + " ".join(live_problems) + ". "
            "For uploads, pick a *Live/Edge* model (I3D visual + log-mel audio).",
            icon="⚠️")

    tab_test, tab_upload, tab_stream = st.tabs(
        ["Select Dataset Sample (MP4)", "Upload New Video", "Connect Stream (RTSP / webcam)"])
    video_path_to_process = None

    normalize = st.toggle(
        "Conform input to the training profile (24 fps, H.264, mono 16 kHz, loudness-normalised)",
        value=True,
        help="Transcodes an arbitrary upload so the snippet window (0.667 s) and audio hop match "
             "the head's training grid, and any container/codec/fps is accepted. It fixes "
             "ingestion compatibility, not content domain shift (footage unlike XD-Violence "
             "still scores low).")

    with tab_test:
        sample_videos = S.list_available_videos()
        if sample_videos:
            video_names = [v.name for v in sample_videos]
            chosen_video_name = st.selectbox("Available Test Video", video_names, index=0)
            video_path_to_process = sample_videos[video_names.index(chosen_video_name)]
        else:
            st.info("No pre-downloaded MP4 samples found in `data/raw/`.")

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Video File (.mp4, .avi, .mkv, .mov)", type=["mp4", "avi", "mkv", "mov"])
        if uploaded_file is not None:
            tmp_dir = Path("data/raw/uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            uploaded_path = tmp_dir / uploaded_file.name
            uploaded_path.write_bytes(uploaded_file.getvalue())
            video_path_to_process = uploaded_path
            st.success(f"Video loaded: `{uploaded_file.name}`")

    with tab_stream:
        st.caption(
            "RTSP URL (`rtsp://cam/ds1`), a local camera (`/dev/video0`) or a local file (looped "
            "as a stream for the demo). The stream is **captured** for the chosen window - video "
            "and audio, synchronised - and then analysed by the same tested pipeline as uploads.")
        stream_source = st.text_input(
            "Stream source", value="", placeholder="rtsp://192.168.1.50:554/stream")
        stream_seconds = st.slider("Capture window (s)", 5, 30, 10, step=5)
        if st.button("📡 Capture Stream Window", disabled=not stream_source):
            try:
                with st.spinner(f"Capturing {stream_seconds}s from the stream…"):
                    video_path_to_process = S.capture_stream(stream_source, stream_seconds)
                st.success(f"Captured `{video_path_to_process.name}` - analyse it below.")
            except RuntimeError as err:
                st.error(f"Stream capture failed: {err}")

    if video_path_to_process is None:
        st.warning("Please select or upload a video to continue.")
        return

    if st.button("⚡ Analyze Video Live", type="primary", use_container_width=True):
        with st.spinner("Extracting audio-visual streams & running multimodal inference…"):
            try:
                res = S.analyze_video(
                    video_path=video_path_to_process,
                    ckpt_path=ckpt_path,
                    threshold=options["threshold"],
                    min_seconds=options["min_seconds"],
                    max_gap_seconds=options["max_gap_seconds"],
                    out_dir=options["sheets_dir"],
                    generate_media=True,
                    adaptive=options["adaptive"],
                    causal=options["causal"],
                    normalize=normalize,
                )
                st.session_state["live_result"] = res
            except Exception as e:
                st.error(f"Analysis error: {e}")

    result = st.session_state.get("live_result")
    if result and result.get("clip_id") == video_path_to_process.stem:
        st.divider()
        clip_id = result["clip_id"]
        timeline = result["timeline"]
        alert = result["alert"]

        m_cols = st.columns(4)
        m_cols[0].metric("Anomaly Score (top-k)", f"{result['clip_score']:.3f}")
        m_cols[1].metric("Analyzed Duration", f"{result['duration_s']:.1f} s")
        m_cols[2].metric("Detected Segments", len(result["segments"]))
        top_cat = alert.get("category", {}).get("label") or "n/a"
        m_cols[3].metric("Dominant Category", top_cat)

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Synchronized Video Playback")
            st.video(str(video_path_to_process))
            st.subheader("Temporal Probability Curve")
            st.plotly_chart(F.timeline_figure(timeline), width="stretch")
            st.caption(
                f"{timeline['n_snippets']} snippets · grid {result['stride_frames']:g} "
                f"frames/snippet · FPS {result['fps']:.1f}")

        with right:
            st.subheader("Incident Sheet & Evidence")
            render_alert(alert)
            render_operator(clip_id, selected_ckpt, timeline, options["decisions_path"])

        st.divider()
        render_decisions(options["decisions_path"])


def render_dataset_review(options: dict) -> None:
    """Offline supervision over pre-scored dataset runs."""
    run = options["run"]
    scores, ground_truth = cached_predictions(str(run.path))
    if not scores:
        st.error(f"`{run.path}/predictions.npz` is empty.")
        return
    decisions_path = options["decisions_path"]
    decided = S.decided_clips(decisions_path)
    progress = S.review_progress(scores, decisions_path)

    header = st.columns(4)
    header[0].metric("Clips Reviewed", f"{progress['n_handled']} / {progress['n_clips']}")
    header[1].metric("Confirmed", progress["by_decision"]["confirmed"])
    header[2].metric("Rejected", progress["by_decision"]["rejected"])
    header[3].metric("Unsure", progress["by_decision"]["unsure"])

    queue = S.review_queue(scores, decided=set(decided), order=options["order"])
    st.divider()
    left, right = st.columns([3, 2])

    with left:
        search = st.text_input("Search clip (substring)", "",
                               placeholder="e.g. Bad.Boys or #00-06-42")
        choices = [row["clip_id"] for row in queue]
        if search:
            choices = [clip for clip in choices if search.lower() in clip.lower()]
        if not choices:
            st.warning("No clips match the search query.")
            return
        clip_id = st.selectbox(
            f"Review Queue ({len(choices)} clips, unreviewed first)",
            choices[:500], index=0,
            format_func=lambda cid: (f"✔ {cid}" if cid in decided else cid))

        intervals = cached_intervals(options["intervals_path"]).get(clip_id, [])
        alert_pre = S.load_alert(options["sheets_dir"], clip_id)
        top_cat = alert_pre.get("category", {}).get("label") if alert_pre else None
        timeline = S.timeline(
            clip_id, scores[clip_id], run.stride_frames, fps=24.0,
            ground_truth=ground_truth.get(clip_id),
            gt_intervals=intervals if options["show_gt"] else None,
            threshold=options["threshold"], min_seconds=options["min_seconds"],
            max_gap_seconds=options["max_gap_seconds"],
            category=top_cat, adaptive=options["adaptive"],
            causal=options["causal"])
        st.plotly_chart(F.timeline_figure(timeline), width="stretch")
        st.caption(
            f"Clip score (mean top-{S.DEFAULT_TOP_K}) **{timeline['clip_score']:.3f}** · "
            f"{timeline['n_snippets']} snippets × {timeline['stride_seconds']:.2f} s = "
            f"{timeline['duration_s']:.0f} s · {len(timeline['segments'])} segment(s) at threshold "
            f"{timeline['threshold']:g} (min duration {options['min_seconds']:g} s)")
        if options["show_gt"]:
            st.info("The green band is test split **ground truth** (evaluation reference); "
                    "the orange bands are **predicted** segments.",
                    icon="ℹ️")

    with right:
        alert = S.load_alert(options["sheets_dir"], clip_id)
        if alert:
            st.subheader("Incident Sheet")
            render_alert(alert)
        else:
            st.subheader("Incident Sheet")
            st.warning("No incident sheet for this clip.", icon="📄")
            st.caption("Generate it and refresh the page:")
            st.code(f".venv/bin/python scripts/incident_sheet.py "
                    f"--ckpt {run.path}/ckpt_best.pt --clip {clip_id} --media", language="bash")
            st.caption("The incident sheet is produced from the checkpoint and clip "
                       "without ground truth — it operates on raw, unannotated videos.")
        render_operator(clip_id, run.name, timeline, decisions_path)

    st.divider()
    render_decisions(decisions_path)


def main() -> None:
    st.set_page_config(page_title="SafeWatch — Supervision & Edge Demo", layout="wide")
    options = sidebar()
    if options["mode"] == "Live Demo / Import Video":
        render_live_demo(options)
    else:
        render_dataset_review(options)


if __name__ == "__main__":      # `streamlit run app.py` sets this; imports do not
    main()
