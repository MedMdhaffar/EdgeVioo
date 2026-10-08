"""P6 - state behind the supervision console: what to show, and what the operator decided.

Deliberately free of Streamlit, torch and network access: the console must stay a *renderer*, so
the same logic can be tested headlessly today and served by FastAPI tomorrow. Everything here
works from artefacts the pipeline already writes:

* ``runs/<run>/predictions.npz`` - per-clip snippet score curves + snippet ground truth
* ``runs/<run>/test_metrics.json`` - the run's metrics and, crucially, its ``stride_frames``
* ``sheets/<clip>_incident.json`` - the alert card built by ``scripts/incident_sheet.py``
* ``data/annotations/test_intervals.csv`` - incident intervals, for the review timeline

The operator's decisions are appended to a **JSONL audit log** (one line per decision, never
rewritten): a supervision tool that overwrites its own trail cannot be audited, and a rejected
alert must remain visible. Each line carries enough context (score, segments, run) to be read years
later without the git history.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from safewatch.eval import metrics as M

#: Decisions an operator can record. ``unsure`` is deliberately kept: forcing a binary choice on a
#: genuinely ambiguous clip is how labels get corrupted.
DECISIONS = ("confirmed", "rejected", "unsure")

DEFAULT_ALERT_THRESHOLD = 0.5
DEFAULT_MIN_SECONDS = 5.333
DEFAULT_MAX_GAP_SECONDS = 2.667
DEFAULT_TOP_K = 5          # must match the training objective's top-k rule (safewatch.losses.mil)

#: The deployed visual extractor. A **trained** model is mandatory: scoring with a randomly
#: initialised MiniRGB is exactly how the live/import path silently produced noise before (the ONNX
#: was exported from a fresh model). Kept as module constants so tests can inject a fixture.
EXTRACTOR_ONNX = Path("export/onnx/minirgb_w24_fp32.onnx")
EXTRACTOR_WEIGHTS = Path("export/onnx/minirgb_w24_distilled.pt")


@dataclass
class RunInfo:
    """One scored run, as the console needs it (no checkpoint loaded)."""

    name: str
    path: Path
    stride_frames: float
    n_clips: int = 0
    ap_global: float | None = None
    ap_per_clip: float | None = None
    macro_per_class_ap: float | None = None
    has_alerts: bool = False
    fusion_mode: str = ""

    @property
    def label(self) -> str:
        ap = "n/a" if self.ap_global is None else f"{self.ap_global:.4f}"
        return f"{self.name}  (APglob {ap})"


def read_json(path: Path, default=None):
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


CHAMPION_RUNS = {
    "2026-10-01_p3edge_i3dmel": "🎥 Live/Edge — I3D visual + log-mel (73.7% AP; use for uploads)",
    "2026-09-29_p3full_cross": "⭐ High-Accuracy Cross-Attention (83.7% AP, Top Performer)",
    "2026-09-29_p3vggish30ep_cross": "⭐ Edge AI Fast CPU (77.5% AP, 6.5ms ONNX)",
    "2026-09-29_p3full_late_clipnorm": "⭐ Late Fusion Normalized (81.4% AP, Independent Heads)",
}


def list_runs(
    runs_dir: str | Path = "runs",
    alerts_dir: str | Path = "sheets",
    champions_only: bool = False,
) -> list[RunInfo]:
    """Every run that has predictions, prioritizing champion models."""
    runs_dir, alerts_dir = Path(runs_dir), Path(alerts_dir)
    out: list[RunInfo] = []
    candidates = sorted((p for p in runs_dir.glob("*_*") if p.is_dir()), reverse=True)
    champ_keys = list(CHAMPION_RUNS.keys())
    candidates.sort(key=lambda p: champ_keys.index(p.name) if p.name in champ_keys else 999)

    for run_dir in candidates:
        if champions_only and run_dir.name not in CHAMPION_RUNS:
            continue
        if not (run_dir / "predictions.npz").exists():
            continue
        cfg = read_json(run_dir / "config.json", {}) or {}
        metrics = read_json(run_dir / "test_metrics.json", {}) or {}
        ap = metrics.get("ap", {})
        alert_files = sorted(alerts_dir.glob(f"{run_dir.name}*_incident.json")) if (
            alerts_dir.exists()) else []
        out.append(RunInfo(
            name=run_dir.name, path=run_dir,
            stride_frames=float(metrics.get("stride_frames") or 64.0),
            n_clips=int(metrics.get("n_clips") or 0),
            ap_global=ap.get("global__pr_auc"),
            ap_per_clip=ap.get("per_video__average_precision"),
            macro_per_class_ap=metrics.get("macro_per_class_ap"),
            has_alerts=bool(alert_files),
            fusion_mode=str(cfg.get("fusion_mode", "")),
        ))
    if champions_only and not out:
        return list_runs(runs_dir, alerts_dir, champions_only=False)
    return out


def load_predictions(run_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``(scores, snippet_ground_truth)`` per clip, straight from the runner's archive."""
    archive = np.load(Path(run_dir) / "predictions.npz")
    scores = {k[len("scores::"):]: archive[k] for k in archive.files if k.startswith("scores::")}
    gt = {k[len("gt::"):]: archive[k] for k in archive.files if k.startswith("gt::")}
    return scores, gt


def clip_score(scores: np.ndarray, top_k: int = DEFAULT_TOP_K) -> float:
    """Clip-level score, using the training objective's top-k rule.

    A clip is positive if *at least one* moment is anomalous, so the mean of the k highest snippet
    probabilities is the right summary - the same surrogate the MIL loss optimises. A plain mean
    would misrank a long quiet clip with one spike against a short loud one.
    """
    values = np.asarray(scores).ravel()
    if values.size == 0:
        return float("nan")
    k = max(1, min(top_k, values.size))
    return float(np.sort(values)[-k:].mean())


def load_intervals(path: str | Path = "data/annotations/test_intervals.csv"
                   ) -> dict[str, list[tuple[int, int]]]:
    """Frame intervals per clip (the annotation file has no category column - see Session 10)."""
    intervals: dict[str, list[tuple[int, int]]] = {}
    path = Path(path)
    if not path.exists():
        return intervals
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            intervals.setdefault(row["clip_id"], []).append((int(row["start"]), int(row["end"])))
    return intervals


def timeline(clip_id: str, scores: np.ndarray, stride_frames: float, fps: float = 24.0,
             ground_truth: np.ndarray | None = None,
             gt_intervals: list[tuple[int, int]] | None = None,
             threshold: float = DEFAULT_ALERT_THRESHOLD,
             min_seconds: float = DEFAULT_MIN_SECONDS,
             max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
             category: str | None = None,
             adaptive: bool = False,
             causal: bool = False) -> dict:
    """Everything the review timeline draws - all times in **seconds**, never in snippets.

    When ``causal=True``, applies causal exponential moving average (EMA) smoothing to the score
    curve. This is display-only: the model scores were produced by a full-clip forward pass and can
    include future context. It does not simulate causal model inference. When ``adaptive=True``,
    applies category-adaptive thresholds and segment rules.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if causal and scores.size > 0:
        # Causal EMA filter (alpha=0.4, looking strictly backward)
        causal_scores = np.zeros_like(scores)
        causal_scores[0] = scores[0]
        alpha = 0.4
        for i in range(1, len(scores)):
            causal_scores[i] = alpha * scores[i] + (1 - alpha) * causal_scores[i - 1]
        scores = causal_scores

    stride_seconds = stride_frames / fps
    if adaptive and category:
        segments = M.category_adaptive_segments(
            scores, category=category, stride_frames=stride_frames, fps=fps,
            default_threshold=threshold, default_min_seconds=min_seconds,
            default_max_gap_seconds=max_gap_seconds)
    else:
        min_length = M.snippets_for_seconds(min_seconds, stride_frames, fps)
        max_gap = M.snippets_for_seconds(max_gap_seconds, stride_frames, fps, minimum=0)
        segments = M.segments_from_scores(scores, threshold, min_length, max_gap)

    def to_seconds(index: int) -> float:
        return round(index * stride_seconds, 2)

    return {
        "clip_id": clip_id,
        "stride_frames": stride_frames,
        "fps": fps,
        "stride_seconds": round(stride_seconds, 3),
        "n_snippets": int(scores.size),
        "duration_s": round(scores.size * stride_seconds, 2),
        "probability": scores.tolist(),
        "seconds": [to_seconds(i) for i in range(scores.size)],
        "threshold": threshold,
        "causal": causal,
        "causal_smoothing": causal,
        "model_inference_context": "full_clip",
        "category": category,
        "segments": [{"start_s": to_seconds(s), "end_s": to_seconds(e + 1),
                      "peak_probability": round(float(scores[s:e + 1].max()), 4),
                      "start_snippet": s, "end_snippet": e} for s, e in segments],
        "clip_score": clip_score(scores),
        "gt_snippets": (None if ground_truth is None
                        else np.asarray(ground_truth).astype(int).tolist()),
        "gt_intervals": [{"start_s": round(a / fps, 2), "end_s": round(b / fps, 2)}
                         for a, b in (gt_intervals or [])],
    }


def review_queue(scores: dict[str, np.ndarray], decided: set[str] | None = None,
                 order: str = "score_desc", top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Clips ordered for review, undecided first.

    Default order is descending clip score: an operator with limited attention should see the alerts
    the model is most confident about first, because confirming or rejecting those is the most
    informative act. Decided clips stay in the list but sink to the end, so the console can still
    show what was already handled.
    """
    decided = decided or set()
    rows = [{"clip_id": clip_id, "clip_score": clip_score(values, top_k),
             "decided": clip_id in decided, "n_snippets": int(np.asarray(values).size)}
            for clip_id, values in scores.items()]
    if order == "score_desc":
        rows.sort(key=lambda row: (row["decided"], -row["clip_score"]))
    elif order == "score_asc":
        rows.sort(key=lambda row: (row["decided"], row["clip_score"]))
    elif order == "clip_id":
        rows.sort(key=lambda row: (row["decided"], row["clip_id"]))
    else:
        raise ValueError(f"unknown order {order!r}; expected score_desc, score_asc or clip_id")
    return rows


def _causal_prefix_forward(model, kind: str, tensors: tuple, prefix_stride: int = 8,
                           normalize_audio: bool = False
                           ) -> tuple[np.ndarray, dict]:
    """Score growing prefixes without future snippets, matching the offline causal evaluator."""
    import torch

    if prefix_stride < 1:
        raise ValueError("prefix_stride must be >= 1")
    n = int(tensors[0].shape[1])
    if n < 1:
        raise ValueError("causal prefix scoring requires at least one snippet")
    endpoints = list(range(prefix_stride, n, prefix_stride)) + [n]
    endpoint_scores = []
    last_out = None
    with torch.no_grad():
        for end in endpoints:
            prefix = tuple(t[:, :end] if t is not None else None for t in tensors)
            if kind == "fusion":
                visual, audio, mask, reliability = prefix
                if normalize_audio:
                    mean = audio.mean(dim=1, keepdim=True)
                    std = audio.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                    audio = (audio - mean) / std
                out = model(visual, audio, mask, reliability)
            else:
                (features, mask) = prefix
                if normalize_audio:
                    mean = features.mean(dim=1, keepdim=True)
                    std = features.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                    features = (features - mean) / std
                out = model(features, mask)
            # Keep endpoint values on the model device. Calling .cpu() here would synchronize
            # GPU execution once per prefix; transfer all endpoint scores together below.
            endpoint_scores.append(torch.sigmoid(out["snippet_logits"])[0, end - 1])
            last_out = out
    assert last_out is not None
    scores = torch.stack(endpoint_scores).cpu().numpy()
    curve = np.zeros(n, dtype=np.float32)
    for index, (end, score) in enumerate(zip(endpoints, scores)):
        stop = endpoints[index + 1] if index + 1 < len(endpoints) else n + 1
        curve[end - 1:stop - 1] = score
    return curve, last_out



@dataclass
class Decision:
    """One operator verdict, as written to the audit log."""

    clip_id: str
    decision: str                      # one of DECISIONS
    run: str = ""
    operator: str = "operator"
    note: str = ""
    clip_score: float | None = None
    n_segments: int = 0
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(
        timespec="seconds"))

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}; expected one of {DECISIONS}")


def load_decisions(path: str | Path) -> list[Decision]:
    """Read the JSONL audit log; empty list when the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[Decision] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Decision(**json.loads(line)))
    return out


def decided_clips(path: str | Path) -> dict[str, str]:
    """``{clip_id: latest decision}`` - the log is append-only, so the LAST line wins."""
    latest: dict[str, str] = {}
    for decision in load_decisions(path):
        latest[decision.clip_id] = decision.decision
    return latest


def record_decision(path: str | Path, decision: Decision) -> None:
    """Append one decision. Never rewrites the file: a rejected alert must stay visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")


def decision_summary(path: str | Path) -> dict:
    """Counts per decision on the *latest* verdict per clip (not per log line)."""
    latest = decided_clips(path)
    counts = {name: 0 for name in DECISIONS}
    for verdict in latest.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    return {"n_decided": len(latest), **counts}


def list_alerts(alerts_dir: str | Path = "sheets") -> dict[str, Path]:
    """``{clip_id: alert json path}`` for every incident sheet on disk."""
    alerts_dir = Path(alerts_dir)
    out: dict[str, Path] = {}
    if not alerts_dir.exists():
        return out
    for path in sorted(alerts_dir.glob("*_incident.json")):
        clip_id = path.name[: -len("_incident.json")]
        out[clip_id] = path
    return out


def load_alert(alerts_dir: str | Path, clip_id: str) -> dict | None:
    """The alert card for one clip, or ``None`` (the console then shows the timeline alone)."""
    path = Path(alerts_dir) / f"{clip_id}_incident.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def review_progress(scores: dict[str, np.ndarray], decisions_path: str | Path) -> dict:
    """How much of the review queue has been handled - the console's headline counter."""
    latest = decided_clips(decisions_path)
    total = len(scores)
    handled = sum(1 for clip_id in scores if clip_id in latest)
    return {"n_clips": total, "n_handled": handled,
            "fraction": round(handled / total, 4) if total else 0.0,
            "by_decision": decision_summary(decisions_path)}


def list_checkpoints(runs_dir: str | Path = "runs", champions_only: bool = False) -> list[Path]:
    """Return all valid ckpt_best.pt / ckpt_last.pt in the runs directory, champions first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    ckpts = sorted(runs_dir.glob("*/ckpt_best.pt"), reverse=True)
    if not ckpts:
        ckpts = sorted(runs_dir.glob("*/*.pt"), reverse=True)
    champ_keys = list(CHAMPION_RUNS.keys())
    ckpts.sort(
        key=lambda p: champ_keys.index(p.parent.name) if p.parent.name in champ_keys else 999
    )
    if champions_only:
        filtered = [p for p in ckpts if p.parent.name in CHAMPION_RUNS]
        if filtered:
            return filtered
    return ckpts


#: Feature spaces the on-the-fly extractor can reproduce, and the modules that produce them.
LIVE_VISUAL_FEATURE_SET = "i3d_official"   # MiniRGB is distilled from I3D (scripts/edge_distill.py)
LIVE_AUDIO_MARKER = "mel"                  # live audio is log-mel patch statistics


def live_feature_support(cfg: dict) -> tuple[bool, list[str]]:
    """Can the live extractor reproduce the input features this checkpoint was trained on?

    The upload path synthesises visual features with the I3D-distilled ``MiniRGB`` CNN and audio
    with log-mel patch statistics. A checkpoint trained on any *other* space (VideoSwin visual, or
    VGGish audio) receives out-of-distribution inputs, which shows up as near-zero anomaly scores on
    obvious incidents (measured: the same clip scores 1.000 on its precomputed features but ~0.001
    live). The console must say so rather than present the number as if it meant something.
    """
    problems = []
    uses_audio = bool(cfg.get("fusion_mode")) or cfg.get("modality") in ("audio", "both")
    uses_visual = bool(cfg.get("fusion_mode")) or cfg.get("modality") in ("visual", "both", None)
    feature_set = cfg.get("feature_set")
    if uses_visual and feature_set != LIVE_VISUAL_FEATURE_SET:
        problems.append(
            f"visual features are {feature_set or 'unknown'} "
            f"(the live extractor only reproduces {LIVE_VISUAL_FEATURE_SET})")
    audio_dir = str(cfg.get("audio_dir") or "")
    if uses_audio and LIVE_AUDIO_MARKER not in audio_dir:
        problems.append(
            f"audio features are {audio_dir or 'unknown'} "
            f"(the live extractor only reproduces {LIVE_AUDIO_MARKER})")
    return (not problems), problems


def list_available_videos(video_root: str | Path = "data/raw/xd-violence/data/video") -> list[Path]:
    """Return all available .mp4 test videos found on disk."""
    root = Path(video_root)
    if not root.exists():
        return []
    candidates = sorted(root.glob("test_videos/*.mp4"))
    if not candidates:
        candidates = sorted(root.glob("*.mp4"))
    if not candidates:
        candidates = sorted(root.glob("**/*.mp4"))
    return candidates


def capture_stream(source: str, seconds: float,
                   out_dir: str | Path = "data/raw/stream_captures") -> Path:
    """Capture ``seconds`` of a stream source (RTSP URL, /dev/video0, or a looped file).

    The subject asks to "import a video or connect a stream". A live stream has no seek, so instead
    of a second stateful streaming scorer, the capture is a plain file (video + audio, synchronised
    by one ffmpeg process) that the same tested ``analyze_video`` path then runs on - one inference
    path, no divergence. The capture profile matches the training profile of the live/edge
    checkpoints, so no re-conformation is needed afterwards.
    """
    from safewatch.edge import stream as stream_mod

    out_dir = Path(out_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = out_dir / f"stream_{stamp}.mp4"
    return stream_mod.capture_stream(source, out, seconds)


def _precomputed_item(cfg: dict, clip_id: str, causal_audio_norm: bool = False):
    """The exact dataset item for a benchmark clip, or ``None`` when it has no precomputed features.

    Reuses :class:`~safewatch.data.multimodal.MultimodalClipDataset` - the same construction the
    scorer (:mod:`safewatch.eval.runner`) and ``scripts/incident_sheet.py`` use - so reviewing a
    benchmark clip in the console reproduces the run's own predictions instead of an approximation
    of them. Arbitrary uploads are not in the list, so they fall through to on-the-fly extraction.
    """
    from safewatch.data.dataset import load_rows
    from safewatch.data.multimodal import MultimodalClipDataset
    from safewatch.eval.runner import STRIDE_FRAMES

    test_csv = Path(cfg.get("lists_dir") or "data/lists") / "test.csv"
    if not test_csv.exists():
        return None
    rows = load_rows(test_csv)
    if clip_id not in {row["clip_id"] for row in rows}:
        return None
    dataset = MultimodalClipDataset(
        test_csv, cfg.get("audio_dir", "data/features/mel"), modality="both",
        audio_stats=cfg.get("audio_stats"), quality_dir=cfg.get("quality_dir"),
        quality_stats=cfg.get("quality_stats"),
        audio_norm=("none" if causal_audio_norm else cfg.get("audio_norm", "none")),
        audio_grid=cfg.get("audio_grid", "patch"),
        stride_frames=STRIDE_FRAMES.get(cfg.get("feature_set")),
        max_snippets=None, train=False, crop=cfg.get("crop", "mean"))
    for index, row in enumerate(dataset.rows):
        if row["clip_id"] == clip_id:
            return dataset[index]
    return None


def analyze_video(
    video_path: str | Path,
    ckpt_path: str | Path,
    threshold: float = DEFAULT_ALERT_THRESHOLD,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    top_k_evidence: int = 5,
    language: str = "fr",
    out_dir: str | Path | None = None,
    generate_media: bool = True,
    adaptive: bool = True,
    causal: bool = False,
    causal_prefix_stride: int = 8,
    normalize: bool = False,
) -> dict:
    """Run end-to-end inference on a video (raw MP4 or benchmark clip) using a checkpoint.

    ``normalize=True`` conforms a non-benchmark input to the training profile (24 fps, H.264, mono
    16 kHz audio, loudness-normalised) before extracting features, so any container/codec/fps is
    accepted and the snippet time base matches the head's training grid. With ``causal=True``, the
    model scores growing prefixes at ``causal_prefix_stride`` and holds each score forward. Feature
    extraction still finishes before model scoring; this is causal inference simulation, not a live
    frame-by-frame decoder. It does not change the content domain.
    """
    import torch

    from safewatch.data import reliability as R
    from safewatch.data.dataset import CATEGORIES
    from safewatch.data.multimodal import load_audio_stats, patches_per_snippet, pool_audio
    from safewatch.edge import aed
    from safewatch.edge.audio import (
        decode_audio,
        detect_acoustic_events,
        mel_patch_stats,
        quality_frames,
    )
    from safewatch.edge.extractor import DEFAULT_INPUT, MiniRGB, frame_grid
    from safewatch.edge.video import probe, sample_frames
    from safewatch.eval import calibration as C
    from safewatch.eval.runner import load_model, resolve_stride
    from safewatch.explain import media as media_mod
    from safewatch.explain import report as report_mod

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    clip_id = video_path.stem
    info = probe(video_path)
    fps = float(info.get("fps") or 24.0)
    frame_count = int(info.get("frames") or 0)

    model, cfg, kind = load_model(str(ckpt_path), torch.device("cpu"))
    stride = float(cfg.get("stride_frames") or 0.0)
    if stride <= 0:
        try:
            stride = resolve_stride(cfg, None)
        except (SystemExit, Exception):
            stride = 16.0
    visual_dim = int(
        cfg.get("visual_dim") or getattr(model, "visual_dim", getattr(model, "in_dim", 1024))
    )
    # An audio-only MIL checkpoint records its width as ``in_dim`` (132 for log-mel), not
    # ``audio_dim``; falling back to 128 silently truncates the mel features and the model then
    # rejects the input (measured: "expected 132 channels, but got 128").
    audio_dim = int(
        cfg.get("audio_dim") or getattr(model, "audio_dim", 0)
        or (cfg.get("in_dim") if cfg.get("modality") == "audio" else 0) or 128
    )

    # 1. Features. Benchmark clips already carry precomputed features on disk; for those, rebuild
    #    the exact dataset item the scorer and scripts/incident_sheet.py use, so the console
    #    reproduces the run's own predictions instead of approximating them with the live extractor.
    #    Everything else (arbitrary uploads) is extracted on the fly from the raw video.
    #    The 0.96 s audio/quality patches must be pooled onto the *same* visual grid the head was
    #    trained on (mirror 64-frame ~2.78 patches/snippet, official 16-frame ~0.69); using one
    #    constant for both slides the audio track off the video by up to 4x.
    pps = patches_per_snippet(stride, fps)
    acoustic_events = []
    aed_tags = None
    quality_arr = None
    reliability_pres = None
    mask_np = None
    prefix_audio_norm = bool(causal and cfg.get("audio_norm") == "clip")
    item = _precomputed_item(cfg, clip_id, causal_audio_norm=prefix_audio_norm)
    if item is not None:
        precomputed = True
        visual_arr = item["visual"].numpy()
        audio_arr = item["audio"].numpy()
        n_snippets = int(item["mask"].numel())
        mask_np = item["mask"].numpy().astype(bool)
        if "reliability" in item:
            reliability_pres = item["reliability"].numpy()
        # named sound events for benchmark clips: the offline YAMNet tags, if extracted
        aed_npz = aed.TAGS_DIR / f"{clip_id}.npz"
        if aed_npz.exists():
            candidate = np.asarray(np.load(aed_npz)["tags"], dtype=np.float32)
            if candidate.shape[0] == n_snippets:
                aed_tags = candidate
    else:
        precomputed = False
        if normalize:
            # Conform a non-benchmark upload to the training profile before extracting: arbitrary
            # fps/codec/sample-rate otherwise shifts the snippet window and the audio hop.
            from safewatch.edge.normalize import normalize_video

            video_path = normalize_video(video_path)
            info = probe(video_path)
            fps = float(info.get("fps") or 24.0)
            frame_count = int(info.get("frames") or 0)
            pps = patches_per_snippet(stride, fps)
        n_snippets = max(1, frame_count // int(stride))
        visual_arr = None

        # Audio extraction & robust standardization
        waveform = decode_audio(video_path)
        raw_mel = mel_patch_stats(waveform)
        quality_arr = quality_frames(waveform)
        acoustic_events = detect_acoustic_events(waveform)
        # named sound events on the fly (frozen YAMNet, ~ms per 0.96 s frame on CPU); degrades to
        # signal statistics when the model artefacts are absent
        if aed.yamnet_available() and waveform.size > 0:
            aed_frames = aed.yamnet_frames(waveform)
            aed_tags = aed.snippet_tags(aed_frames, n_snippets, snippet_s=stride / fps)

        if raw_mel.shape[0] > 0:
            audio_arr = pool_audio(raw_mel, n_snippets, pps)
            stats_file = cfg.get("audio_stats")
            audio_stats = load_audio_stats(stats_file) if stats_file else None
            if audio_stats is not None and audio_arr.shape[1] == audio_stats[0].shape[0]:
                audio_arr = (audio_arr - audio_stats[0]) / np.maximum(audio_stats[1], 1e-6)
            if cfg.get("audio_norm") == "clip" or audio_stats is None:
                if causal:
                    prefix_audio_norm = True
                else:
                    a_mean = audio_arr.mean(axis=0, keepdims=True)
                    a_std = np.maximum(audio_arr.std(axis=0, keepdims=True), 1e-4)
                    audio_arr = (audio_arr - a_mean) / a_std
        else:
            audio_arr = np.zeros((n_snippets, audio_dim), dtype=np.float32)

        if audio_arr.shape[1] != audio_dim:
            if audio_arr.shape[1] > audio_dim:
                audio_arr = audio_arr[:, :audio_dim]
            else:
                audio_arr = np.pad(audio_arr, ((0, 0), (0, audio_dim - audio_arr.shape[1])))

        # Visual extraction
        if visual_arr is None:
            indices = frame_grid(frame_count, n_snippets, snippet_frames=int(stride))
            sampled_frames = sample_frames(video_path, indices, size=DEFAULT_INPUT)
            # ImageNet standard normalization expected by distilled backbone
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
            norm_frames = (sampled_frames - mean) / std
            frames_t = torch.from_numpy(norm_frames).permute(0, 3, 1, 2).unsqueeze(0)

            # No motion weighting: the distilled backbone was trained on the plain ImageNet-
            # normalised frames (scripts/edge_distill.py), so multiplying its output by a motion
            # gain would feed the head a feature it never saw during distillation.
            # The extractor MUST be trained. The ONNX path is preferred (it is what the edge
            # budget prices, parity 1e-8); the .pt is the fallback when the ONNX is missing. A
            # fresh MiniRGB is NEVER acceptable: that is exactly how the live path silently
            # scored noise before (the ONNX had been exported from a random init).
            onnx_backbone = EXTRACTOR_ONNX
            distilled = EXTRACTOR_WEIGHTS
            if onnx_backbone.exists():
                import onnxruntime as ort
                sess = ort.InferenceSession(str(onnx_backbone), providers=["CPUExecutionProvider"])
                vis_extracted = sess.run(None, {"frames": frames_t.squeeze(0).numpy()})[0]
            elif distilled.exists():
                ckpt = torch.load(distilled, map_location="cpu", weights_only=False)
                mini_net = MiniRGB(feature_dim=1024, width=int(ckpt["width"]))
                mini_net.load_state_dict(ckpt["model"])
                mini_net.eval()
                with torch.no_grad():
                    # squeeze the batch: the ONNX graph takes (T, 3, H, W), so this path must too
                    vis_extracted = mini_net(frames_t.squeeze(0)).numpy()
            else:
                raise RuntimeError(
                    "no trained MiniRGB extractor found (expected "
                    "export/onnx/minirgb_w24_fp32.onnx or minirgb_w24_distilled.pt); refusing to "
                    "score with a randomly-initialised network - run scripts/run_live_edge.sh")
            if visual_dim <= vis_extracted.shape[1]:
                visual_arr = vis_extracted[:, :visual_dim]
            else:
                diff = visual_dim - vis_extracted.shape[1]
                visual_arr = np.pad(vis_extracted, ((0, 0), (0, diff)))

    # 3. Reliability features - the gate must see the SAME standardised, same-grid channels it was
    #    trained on: raw per-patch quality pooled to the snippet grid and z-scored with the
    #    checkpoint's train-split stats (feed it pooled-then-pooled or unstandardised and the gate
    #    receives a distribution it never learned). Precomputed clips carry these directly.
    reliability_arr = reliability_pres
    n_rel = int(cfg.get("n_reliability", 0))
    if reliability_arr is None and n_rel > 0:
        quality_stats = R.load_quality_stats(cfg.get("quality_stats"))
        reliability_arr = R.reliability_channels(
            quality_arr, visual_arr, n_snippets, quality_stats, pps)
    if reliability_arr is not None and reliability_arr.shape[-1] != n_rel:
        if reliability_arr.shape[-1] > n_rel:
            reliability_arr = reliability_arr[:, :n_rel]
        else:
            diff = n_rel - reliability_arr.shape[-1]
            reliability_arr = np.pad(reliability_arr, ((0, 0), (0, diff)))

    # 4. Model forward pass
    visual_t = torch.from_numpy(visual_arr).unsqueeze(0).float()
    audio_t = torch.from_numpy(audio_arr).unsqueeze(0).float()
    mask_t = (torch.from_numpy(mask_np).unsqueeze(0) if mask_np is not None
              else torch.ones((1, n_snippets), dtype=torch.bool))
    rel_t = (torch.from_numpy(reliability_arr).unsqueeze(0).float()
             if reliability_arr is not None else None)

    meta = {
        "inference": {
            "mode": "causal_prefix" if causal else "full_clip",
            "prefix_stride": int(causal_prefix_stride) if causal else None,
            "note": ("Each prefix uses only snippets up to its endpoint; feature extraction still "
                     "finishes before model scoring." if causal else
                     "The model scores the full clip and can use future context."),
        },
        "provenance": {
            "checkpoint": str(ckpt_path),
            "clip_id": clip_id,
            "stride_frames": stride,
            "fusion_mode": cfg.get("fusion_mode", "mil"),
            "inference_mode": (f"causal_prefix_stride_{causal_prefix_stride}" if causal
                               else "full_clip"),
            "feature_extraction_before_scoring": True,
        },
        "acoustic_events": acoustic_events,
    }

    causal_probability = None
    with torch.no_grad():
        if kind == "fusion":
            if causal:
                causal_probability, out = _causal_prefix_forward(
                    model, kind, (visual_t, audio_t, mask_t, rel_t), causal_prefix_stride,
                    normalize_audio=prefix_audio_norm)
            else:
                out = model(visual_t, audio_t, mask_t, rel_t)
            v_drop, a_drop, r_drop = R.drop_modality(visual_t, audio_t, rel_t, "audio")
            out_no_audio = model(v_drop, a_drop, mask_t, r_drop)
            v_drop, a_drop, r_drop = R.drop_modality(visual_t, audio_t, rel_t, "visual")
            out_no_visual = model(v_drop, a_drop, mask_t, r_drop)

            intact_s = torch.sigmoid(out["snippet_logits"])[0].numpy()
            without_a = torch.sigmoid(out_no_audio["snippet_logits"])[0].numpy()
            without_v = torch.sigmoid(out_no_visual["snippet_logits"])[0].numpy()
            meta["drop_one"] = {
                "audio": {
                    "mean_probability_intact": round(float(intact_s.mean()), 4),
                    "mean_probability_without": round(float(without_a.mean()), 4),
                    "delta": round(float(intact_s.mean() - without_a.mean()), 4),
                },
                "visual": {
                    "mean_probability_intact": round(float(intact_s.mean()), 4),
                    "mean_probability_without": round(float(without_v.mean()), 4),
                    "delta": round(float(intact_s.mean() - without_v.mean()), 4),
                },
            }
        else:
            feat_t = audio_t if cfg.get("modality") == "audio" else visual_t
            if causal:
                causal_probability, out = _causal_prefix_forward(
                    model, kind, (feat_t, mask_t), causal_prefix_stride,
                    normalize_audio=(prefix_audio_norm and cfg.get("modality") == "audio"))
            else:
                out = model(feat_t, mask_t)

    snippet_probability = (causal_probability if causal_probability is not None else
                           torch.sigmoid(out["snippet_logits"])[0].numpy())
    attention = out.get("attn")
    alpha = out.get("alpha")
    snippet_visual = (torch.sigmoid(out["snippet_visual"])[0].numpy()
                      if out.get("snippet_visual") is not None else None)
    snippet_audio = (torch.sigmoid(out["snippet_audio"])[0].numpy()
                     if out.get("snippet_audio") is not None else None)

    category_probability = {
        name: float(torch.sigmoid(out["multi_logits"])[0][index])
        for index, name in enumerate(CATEGORIES)
    } if "multi_logits" in out else {}

    context = report_mod.ClipContext(
        clip_id=clip_id,
        stride_frames=float(stride),
        fps=fps,
        n_snippets=n_snippets,
        snippet_probability=snippet_probability,
        attention=attention.numpy()[0] if attention is not None else None,
        alpha=alpha.numpy()[0] if alpha is not None else None,
        snippet_visual=snippet_visual,
        snippet_audio=snippet_audio,
        category_probability=category_probability,
        reliability=reliability_arr,
        video_path=str(video_path),
        meta=meta,
        aed_tags=aed_tags,
    )

    alert_config = report_mod.AlertConfig(
        threshold=threshold,
        min_seconds=min_seconds,
        max_gap_seconds=max_gap_seconds,
        top_k_evidence=top_k_evidence,
        language=language,
        category_adaptive=adaptive,
    )

    calib_file = Path(ckpt_path).parent / "calibration.json"
    calibration = C.CalibrationMap.load(calib_file) if calib_file.exists() else None
    alert_card = report_mod.build_alert(context, alert_config, calibration)

    top_cat = None
    if category_probability:
        ranked = sorted(category_probability.items(), key=lambda kv: kv[1], reverse=True)
        if ranked and ranked[0][1] >= 0.15:
            top_cat = ranked[0][0]

    timeline_dict = timeline(
        clip_id=clip_id,
        scores=snippet_probability,
        stride_frames=stride,
        fps=fps,
        threshold=threshold,
        min_seconds=min_seconds,
        max_gap_seconds=max_gap_seconds,
        category=top_cat,
        adaptive=adaptive,
        # The model curve above is already prefix-causal; this flag is only the review-curve EMA.
        causal=False,
    )
    timeline_dict["causal_inference"] = causal
    timeline_dict["causal_prefix_stride"] = int(causal_prefix_stride) if causal else None
    timeline_dict["model_inference_context"] = (
        f"causal_prefix_stride_{causal_prefix_stride}" if causal else "full_clip")

    # 6. Optional media rendering
    if generate_media and out_dir is not None:
        out_dir = Path(out_dir)
        frames_dir = out_dir / "frames"
        ev = alert_card.get("visual_evidence", {}).get("top_snippets", [])
        snippets = [s["snippet"] for s in ev][:top_k_evidence]
        if snippets:
            times = media_mod.frame_times(snippets, stride, fps)
            keyframes = media_mod.key_frames(video_path, times, frames_dir, prefix=clip_id)
            alert_card.setdefault("media", {})["key_frames"] = keyframes

        if alert_card.get("segments"):
            strongest = max(alert_card["segments"], key=lambda s: s["peak_probability"])
            spec_path = media_mod.spectrogram(
                video_path,
                out_dir / f"{clip_id}_spectrogram.png",
                start_s=strongest["start_s"],
                end_s=strongest["end_s"],
                title=f"{clip_id} [{strongest['start_s']:.1f}-{strongest['end_s']:.1f} s]",
            )
            if spec_path:
                alert_card.setdefault("media", {})["spectrogram"] = str(spec_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = json.dumps(alert_card, indent=2)
        (out_dir / f"{clip_id}_incident.json").write_text(out_json, encoding="utf-8")

    return {
        "clip_id": clip_id,
        "video_path": str(video_path),
        "timeline": timeline_dict,
        "alert": alert_card,
        "context": context,
        "stride_frames": stride,
        "fps": fps,
        "duration_s": timeline_dict["duration_s"],
        "clip_score": timeline_dict["clip_score"],
        "segments": timeline_dict["segments"],
        "precomputed_features": precomputed,
        "feature_support": live_feature_support(cfg),
    }
