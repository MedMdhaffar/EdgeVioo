"""Tests for the P6 supervision console's pure state layer (`safewatch.ui.state`).

The console is a Streamlit script, but everything it *decides* is testable here: the
snippet->seconds conversion (which grid is being shown), the review order, and the append-only
decision log. The log is pinned in particular because it is the project's only operator-facing audit
trail - if it ever started rewriting itself, a rejected alert would silently disappear.

Run:  .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from safewatch.ui import state as S


@pytest.fixture
def trained_extractor(tmp_path, monkeypatch):
    """Inject a small *trained* MiniRGB so ``analyze_video`` never uses a random network.

    The live path now refuses to score without a trained extractor (the deployed ONNX used to be
    exported from a randomly-initialised model), so these tests provide a real one.
    """
    import torch

    from safewatch.edge.extractor import MiniRGB

    weights = tmp_path / "minirgb_distilled.pt"
    torch.save({"model": MiniRGB(feature_dim=1024, width=16).eval().state_dict(), "width": 16},
               weights)
    monkeypatch.setattr(S, "EXTRACTOR_ONNX", tmp_path / "absent.onnx")
    monkeypatch.setattr(S, "EXTRACTOR_WEIGHTS", weights)


def test_timeline_converts_snippets_to_seconds_on_the_run_grid():
    """The same snippet index is 2.67 s on the mirror grid and 0.67 s on the official one."""
    scores = np.ones(20, dtype=np.float32)

    mirror = S.timeline("c", scores, stride_frames=64.0)
    official = S.timeline("c", scores, stride_frames=16.0)

    assert mirror["stride_seconds"] == pytest.approx(64 / 24, rel=1e-3)
    assert official["stride_seconds"] == pytest.approx(16 / 24, rel=1e-3)
    assert mirror["duration_s"] > 3 * official["duration_s"]
    assert mirror["seconds"][1] == pytest.approx(2.67, abs=0.01)
    assert official["seconds"][1] == pytest.approx(0.67, abs=0.01)


def test_timeline_segments_follow_the_seconds_rule_not_the_snippet_count():
    """A 20-snippet burst is 13.3 s on the official grid: it survives a 5.33 s rule."""
    scores = np.zeros(60, dtype=np.float32)
    scores[10:30] = 0.9

    report = S.timeline("c", scores, stride_frames=16.0, threshold=0.5,
                        min_seconds=5.333, max_gap_seconds=2.667)

    assert len(report["segments"]) == 1
    segment = report["segments"][0]
    assert segment["start_s"] == pytest.approx(10 * 16 / 24, abs=0.01)
    assert segment["end_s"] == pytest.approx(30 * 16 / 24, abs=0.01)
    assert segment["peak_probability"] == pytest.approx(0.9, abs=1e-6)

    # a 2-snippet burst (1.33 s) is below the rule and must not be reported as an incident
    short = np.zeros(60, dtype=np.float32)
    short[5:7] = 0.9
    assert S.timeline("c", short, stride_frames=16.0,
                      min_seconds=5.333, max_gap_seconds=2.667)["segments"] == []


def test_timeline_adaptive_and_causal_modes():
    short = np.zeros(60, dtype=np.float32)
    short[5:7] = 0.9

    # Without adaptive: pruned
    std = S.timeline("c", short, stride_frames=16.0, adaptive=False)
    assert std["segments"] == []

    # With adaptive and car_accident: caught
    car = S.timeline("c", short, stride_frames=16.0, category="car_accident", adaptive=True)
    assert len(car["segments"]) == 1

    # Causal mode flag is preserved
    causal_tl = S.timeline("c", short, stride_frames=16.0, causal=True)
    assert causal_tl["causal"] is True


def test_causal_prefix_forward_scores_only_available_prefix_and_holds_forward():
    import torch

    class PrefixEcho(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_lengths = []

        def forward(self, features, mask):
            self.seen_lengths.append(int(features.shape[1]))
            return {"snippet_logits": features[:, :, 0]}

    model = PrefixEcho()
    features = torch.arange(1, 11, dtype=torch.float32).view(1, 10, 1)
    mask = torch.ones((1, 10), dtype=torch.bool)

    curve, final = S._causal_prefix_forward(model, "mil", (features, mask), prefix_stride=4)

    assert model.seen_lengths == [4, 8, 10]
    assert curve[:3].tolist() == [0.0, 0.0, 0.0]
    assert curve[3] == pytest.approx(float(torch.sigmoid(torch.tensor(4.0))))
    assert curve[4:7].tolist() == pytest.approx([curve[3]] * 3)
    assert curve[7] == pytest.approx(float(torch.sigmoid(torch.tensor(8.0))))
    assert curve[8] == pytest.approx(curve[7])
    assert curve[9] == pytest.approx(float(torch.sigmoid(torch.tensor(10.0))))
    assert final["snippet_logits"].shape == (1, 10)


def test_causal_fusion_prefix_normalizes_audio_from_prefix_only():
    import torch

    class FusionAudioEcho(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_lengths = []
            self.normalized_audio = []

        def forward(self, visual, audio, mask, reliability):
            self.seen_lengths.append(int(audio.shape[1]))
            self.normalized_audio.append(audio[0, :, 0].clone())
            return {"snippet_logits": audio[:, :, 0]}

    model = FusionAudioEcho()
    visual = torch.zeros((1, 5, 2))
    audio = torch.tensor([0.0, 2.0, 4.0, 10.0, 14.0]).view(1, 5, 1)
    mask = torch.ones((1, 5), dtype=torch.bool)

    S._causal_prefix_forward(
        model, "fusion", (visual, audio, mask, None), prefix_stride=2, normalize_audio=True)

    assert model.seen_lengths == [2, 4, 5]
    assert model.normalized_audio[0].tolist() == pytest.approx([-1.0, 1.0])
    assert model.normalized_audio[1][-1].item() == pytest.approx(
        (10.0 - 4.0) / np.sqrt(14.0), rel=1e-5)


def test_clip_score_is_the_top_k_mean_matching_the_training_objective():
    scores = np.array([0.1, 0.2, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert S.clip_score(scores, top_k=5) == pytest.approx(1.0)
    quiet_spike = np.array([0.0] * 40 + [1.0], dtype=np.float32)
    loud_clip = np.full(41, 0.6, dtype=np.float32)
    # one spike does not beat a consistently loud clip at top_k=5 (the mean would say the opposite)
    assert S.clip_score(quiet_spike, 5) < S.clip_score(loud_clip, 5)
    assert np.isnan(S.clip_score(np.array([], dtype=np.float32)))     # empty curve -> NaN


def test_review_queue_puts_undecided_clips_first_and_orders_by_score():
    scores = {"low": np.full(10, 0.1, dtype=np.float32),
              "high": np.full(10, 0.9, dtype=np.float32),
              "mid": np.full(10, 0.5, dtype=np.float32)}
    queue = S.review_queue(scores, decided={"high"})
    assert [row["clip_id"] for row in queue] == ["mid", "low", "high"]
    assert queue[0]["decided"] is False and queue[-1]["decided"] is True

    ascending = S.review_queue(scores, order="score_asc")
    assert [row["clip_id"] for row in ascending] == ["low", "mid", "high"]
    with pytest.raises(ValueError):
        S.review_queue(scores, order="nonsense")


def test_decision_log_is_append_only_and_the_last_verdict_wins(tmp_path: Path):
    log = tmp_path / "decisions.jsonl"
    assert S.load_decisions(log) == []                     # no file yet
    S.record_decision(log, S.Decision(clip_id="a", decision="rejected", run="r", note="no event"))
    S.record_decision(log, S.Decision(clip_id="a", decision="confirmed", run="r", note="on review"))
    S.record_decision(log, S.Decision(clip_id="b", decision="unsure", run="r"))

    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3                                 # nothing was overwritten
    assert S.decided_clips(log) == {"a": "confirmed", "b": "unsure"}
    assert S.decision_summary(log) == {"n_decided": 2, "confirmed": 1, "rejected": 0, "unsure": 1}


def test_decision_rejects_an_unknown_verdict():
    with pytest.raises(ValueError):
        S.Decision(clip_id="a", decision="maybe")


def test_review_progress_counts_handled_clips(tmp_path: Path):
    log = tmp_path / "decisions.jsonl"
    S.record_decision(log, S.Decision(clip_id="a", decision="confirmed"))
    scores = {"a": np.zeros(5, dtype=np.float32), "b": np.zeros(5, dtype=np.float32)}
    progress = S.review_progress(scores, log)
    assert progress["n_clips"] == 2 and progress["n_handled"] == 1
    assert progress["fraction"] == pytest.approx(0.5)


def test_load_alert_returns_none_when_no_sheet_exists(tmp_path: Path):
    assert S.load_alert(tmp_path, "unknown_clip") is None
    (tmp_path / "known_clip_incident.json").write_text(json.dumps({"clip_id": "known_clip"}),
                                                      encoding="utf-8")
    assert S.load_alert(tmp_path, "known_clip")["clip_id"] == "known_clip"
    assert set(S.list_alerts(tmp_path)) == {"known_clip"}


def test_list_runs_skips_dirs_without_predictions_and_reads_the_grid(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    good = runs_dir / "2026-09-30_demo_cross"
    good.mkdir(parents=True)
    np.savez_compressed(good / "predictions.npz",
                        **{"scores::c1": np.ones(7, dtype=np.float32),
                           "gt::c1": np.zeros(7, dtype=np.int64)})
    (good / "config.json").write_text(json.dumps({"fusion_mode": "cross"}), encoding="utf-8")
    (good / "test_metrics.json").write_text(json.dumps(
        {"stride_frames": 16.0, "n_clips": 1, "macro_per_class_ap": 0.5,
         "ap": {"global__pr_auc": 0.7, "per_video__average_precision": 0.8}}), encoding="utf-8")
    (runs_dir / "2026-09-30_not_scored").mkdir()                   # no predictions -> skipped

    runs = S.list_runs(runs_dir, alerts_dir=tmp_path / "sheets")
    assert len(runs) == 1
    info = runs[0]
    assert info.name == "2026-09-30_demo_cross" and info.stride_frames == 16.0
    assert info.ap_global == pytest.approx(0.7) and info.fusion_mode == "cross"
    assert "APglob 0.7000" in info.label
    scores, gt = S.load_predictions(good)
    assert set(scores) == {"c1"} and set(gt) == {"c1"}


def test_list_checkpoints_and_videos(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_a = runs_dir / "exp_a"
    run_a.mkdir(parents=True)
    (run_a / "ckpt_best.pt").write_text("fake_ckpt", encoding="utf-8")

    ckpts = S.list_checkpoints(runs_dir)
    assert len(ckpts) == 1
    assert ckpts[0].name == "ckpt_best.pt"

    vdir = tmp_path / "videos" / "test_videos"
    vdir.mkdir(parents=True)
    (vdir / "clip_1.mp4").write_text("fake_mp4", encoding="utf-8")
    videos = S.list_available_videos(tmp_path / "videos")
    assert len(videos) == 1
    assert videos[0].name == "clip_1.mp4"


def test_analyze_video_on_synthetic_clip(tmp_path: Path, trained_extractor):
    cv2 = pytest.importorskip("cv2")
    import torch

    from safewatch.models.mil import MILModel

    # 1. Create a synthetic video
    video_path = tmp_path / "synth_video.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder")
    for i in range(32):
        writer.write(np.full((64, 64, 3), 50 + i, dtype=np.uint8))
    writer.release()

    # 2. Create a small dummy checkpoint
    model = MILModel(in_dim=1024, emb=32, hidden=64)
    run_dir = tmp_path / "runs" / "demo_run"
    run_dir.mkdir(parents=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    cfg = {"in_dim": 1024, "emb": 32, "hidden": 64, "stride_frames": 16.0,
           "feature_set": "i3d_official"}
    torch.save({"model": model.state_dict(), "config": cfg}, str(ckpt_path))

    # 3. Run analyze_video
    sheets_dir = tmp_path / "sheets"
    result = S.analyze_video(
        video_path=video_path,
        ckpt_path=ckpt_path,
        threshold=0.4,
        min_seconds=1.0,
        max_gap_seconds=1.0,
        out_dir=sheets_dir,
        generate_media=True,
    )

    assert result["clip_id"] == "synth_video"
    assert "timeline" in result
    assert "alert" in result
    assert result["timeline"]["n_snippets"] > 0
    assert (sheets_dir / "synth_video_incident.json").exists()

    causal_result = S.analyze_video(
        video_path=video_path,
        ckpt_path=ckpt_path,
        generate_media=False,
        causal=True,
        causal_prefix_stride=1,
    )
    assert causal_result["timeline"]["causal_inference"] is True
    assert causal_result["timeline"]["causal_prefix_stride"] == 1
    assert causal_result["timeline"]["model_inference_context"] == "causal_prefix_stride_1"
    assert causal_result["alert"]["provenance"]["inference_mode"] == "causal_prefix_stride_1"



def test_live_feature_support_flags_out_of_distribution_checkpoints():
    """The upload path reproduces I3D visual + log-mel audio only; anything else must be flagged.

    A checkpoint trained on VideoSwin visual or VGGish audio receives out-of-distribution inputs
    from the live extractor, which is why an obvious crash scored ~0.001 instead of ~1.0. The
    console has to say so instead of showing the number as if it meant something.
    """
    ok, problems = S.live_feature_support(
        {"feature_set": "i3d_official", "audio_dir": "data/features/mel"})
    assert ok and problems == []

    bad, problems = S.live_feature_support(
        {"fusion_mode": "cross", "feature_set": "swin_rgb",
         "audio_dir": "data/features/vggish_snippet"})
    assert not bad
    assert any("swin_rgb" in p for p in problems)
    assert any("vggish" in p for p in problems)

    # A visual-only MIL model has no audio stream to flag.
    ok, problems = S.live_feature_support(
        {"modality": "visual", "feature_set": "i3d_official"})
    assert ok and problems == []


def test_analyze_video_reports_the_feature_space_it_used(tmp_path, trained_extractor):
    """``analyze_video`` must tell the console whether the model's inputs are live-reproducible."""
    import cv2
    import torch

    from safewatch.models.mil import MILModel

    video_path = tmp_path / "grid_video.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder")
    for i in range(32):
        writer.write(np.full((64, 64, 3), 50 + i, dtype=np.uint8))
    writer.release()

    model = MILModel(in_dim=1024, emb=32, hidden=64)
    run_dir = tmp_path / "runs" / "demo_run"
    run_dir.mkdir(parents=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save({"model": model.state_dict(),
                "config": {"in_dim": 1024, "emb": 32, "hidden": 64, "stride_frames": 16.0,
                           "feature_set": "i3d_official"}}, str(ckpt_path))

    result = S.analyze_video(video_path=video_path, ckpt_path=ckpt_path, generate_media=False)
    assert result["precomputed_features"] is False
    assert result["feature_support"] == (True, [])


def test_analyze_video_audio_only_mil_uses_in_dim_width(tmp_path, trained_extractor):
    """An audio-only MIL checkpoint records its width as ``in_dim`` (132), not ``audio_dim`` (128).

    Regression: the live path defaulted to 128 and truncated the mel features, so the model rejected
    its own input ("expected 132 channels, but got 128").
    """
    import cv2
    import torch

    from safewatch.models.mil import MILModel

    video_path = tmp_path / "audio_only.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder")
    for i in range(32):
        writer.write(np.full((64, 64, 3), 50 + i, dtype=np.uint8))
    writer.release()

    model = MILModel(in_dim=132, emb=32, hidden=64)
    run_dir = tmp_path / "runs" / "audio_run"
    run_dir.mkdir(parents=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save({"model": model.state_dict(),
                "config": {"in_dim": 132, "modality": "audio", "emb": 32, "hidden": 64,
                           "stride_frames": 16.0, "feature_set": "swin_rgb",
                           "audio_dir": "data/features/mel"}}, str(ckpt_path))

    result = S.analyze_video(video_path=video_path, ckpt_path=ckpt_path, generate_media=False)
    assert result["feature_support"] == (True, [])           # audio-only mel is live-reproducible
    assert len(result["timeline"]["probability"]) > 0
