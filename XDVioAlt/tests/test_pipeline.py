"""Regression tests for the model/data/scoring plumbing (`test_metrics.py` cannot see these).

Pinned here, because each of them has broken at least once or would break silently:

* the batch key produced by the dataset is exactly ``TrainConfig.feature_key`` -- a visual run once
  died on its first batch with ``KeyError: 'visual'`` when the dataset was a plain ``ClipDataset``
  (key ``"features"``);
* audio patches are pooled onto the snippet grid with the same window as the visual features
  (a misalignment "quietly destroys fusion", see ``safewatch/data/multimodal.py``);
* every fusion mode returns the documented output dict, including ``alpha`` for the adaptive gate;
* the scorer recovers ``in_dim`` from the weights, so ``i3d_official`` (1024-d) checkpoints load.

Run:  .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import csv
import math
import warnings

import numpy as np
import pytest
import torch

from safewatch.data.multimodal import (
    PATCHES_PER_SNIPPET,
    MultimodalClipDataset,
    clip_normalise,
    pool_audio,
)
from safewatch.data.reliability import (
    AUDIO_DIM as AUDIO_QUALITY_DIM,
)
from safewatch.data.reliability import (
    DEGRADED,
    RELIABILITY_DIM,
    drop_modality,
    reliability_channels,
    standardise,
)
from safewatch.eval import metrics as M
from safewatch.eval.runner import (
    IN_DIMS,
    compute_metrics,
    infer_in_dim,
    load_model,
    model_kind,
    resolve_stride,
    score_clips,
)
from safewatch.models.fusion import FUSION_MODES, FusionModel
from safewatch.models.mil import MILModel
from safewatch.train import TrainConfig, _make_dataset

VISUAL_DIM = 6
AUDIO_DIM = 3
N_SNIPPETS = 10
OFFICIAL_DIM = IN_DIMS["i3d_official"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def write_clip(tmp_path, clip_id="clip_a", n_snippets=N_SNIPPETS, visual_dim=VISUAL_DIM,
               audio_dim=AUDIO_DIM, encode_index=False, quality=False):
    """Write one clip's features on disk and return its CSV row (5-crop visual + audio patches).

    Layout follows the extractor: ``<audio_dir>/<clip_id>.npy`` and a ``(T, 5, D)`` visual file
    referenced from the CSV. ``quality=True`` also writes the audio signal-quality stats the
    adaptive gate consumes, in ``<tmp_path>/quality/<clip_id>.npy``.
    """
    rng = np.random.default_rng(0)
    if encode_index:  # every crop of snippet t carries the value t -> crop "mean" == t
        visual = np.repeat(np.arange(n_snippets, dtype=np.float32)[:, None, None], 5, axis=1)
    else:
        visual = rng.random((n_snippets, 5, visual_dim), dtype=np.float32)
    visual_path = tmp_path / f"{clip_id}_visual.npy"
    np.save(visual_path, visual)

    patches = math.ceil(PATCHES_PER_SNIPPET * n_snippets)
    patch_of = np.minimum((np.arange(patches) / PATCHES_PER_SNIPPET).astype(int), n_snippets - 1)
    if encode_index:  # each patch carries the index of the snippet it belongs to
        audio = patch_of[:, None].astype(np.float32)
    else:
        audio = rng.random((patches, audio_dim), dtype=np.float32)
    np.save(tmp_path / f"{clip_id}.npy", audio)   # audio is looked up by clip_id

    if quality:                                    # same 0.96 s grid as the audio features
        quality_dir = tmp_path / "quality"
        quality_dir.mkdir(exist_ok=True)
        if encode_index:
            stats = np.repeat(patch_of[:, None], AUDIO_QUALITY_DIM, axis=1)
        else:
            stats = rng.random((patches, AUDIO_QUALITY_DIM), dtype=np.float32)
        np.save(quality_dir / f"{clip_id}.npy", stats.astype(np.float32))

    return {"clip_id": clip_id, "split": "test", "path": str(visual_path), "T": str(n_snippets),
            "binary": "1", "labels": "fighting"}


def write_csv(tmp_path, rows, name="test.csv"):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def train_config(tmp_path, modality="visual", **overrides):
    """A config that reads features from ``tmp_path`` and skips audio normalisation."""
    return TrainConfig(lists_dir=str(tmp_path), audio_dir=str(tmp_path),
                       audio_stats=str(tmp_path / "missing_stats.json"), max_snippets=4,
                       modality=modality, **overrides)


# --------------------------------------------------------------------------- #
# dataset <-> training loop contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modality", ["visual", "audio"])
def test_make_dataset_batch_keys_match_feature_key(tmp_path, modality):
    """Regression: visual runs died with KeyError 'visual' because ClipDataset yields 'features'
    while the training loop indexes the batch with TrainConfig.feature_key."""
    csv_path = write_csv(tmp_path, [write_clip(tmp_path)])
    cfg = train_config(tmp_path, modality=modality)
    dataset = _make_dataset(cfg, csv_path, train=False)
    item = dataset[0]

    assert cfg.feature_key in item, f"batch keys {sorted(item)} lack {cfg.feature_key!r}"
    expected_dim = VISUAL_DIM if modality == "visual" else AUDIO_DIM
    assert item[cfg.feature_key].shape == (cfg.max_snippets, expected_dim)
    assert item["mask"].shape == (cfg.max_snippets,)


def test_make_dataset_rejects_fusion_modality(tmp_path):
    """modality='both' belongs to safewatch.train_fusion (MILModel takes a single input)."""
    csv_path = write_csv(tmp_path, [write_clip(tmp_path)])
    with pytest.raises(ValueError, match="both"):
        _make_dataset(train_config(tmp_path, modality="both"), csv_path, train=True)


def test_visual_and_audio_share_one_temporal_crop(tmp_path):
    """Both modalities must be cut with the same window, otherwise fusion sees audio shifted by
    seconds with respect to the video snippet it belongs to."""
    row = write_clip(tmp_path, clip_id="clip_align", encode_index=True)
    csv_path = write_csv(tmp_path, [row])
    dataset = MultimodalClipDataset(csv_path, tmp_path, modality="both", audio_stats=None,
                                    max_snippets=4, crop="mean", train=True, seed=3)
    item = dataset[0]

    assert item["visual"].shape[0] == item["audio"].shape[0] == 4
    assert torch.allclose(item["visual"][:, 0], item["audio"][:, 0])


# --------------------------------------------------------------------------- #
# audio patch -> snippet grid
# --------------------------------------------------------------------------- #
def test_pool_audio_follows_the_snippet_grid():
    """Patch p feeds snippet floor(p / PATCHES_PER_SNIPPET), i.e. the band
    [ceil(i*PPS), ceil((i+1)*PPS)); each snippet is the mean of its own patches."""
    n_snippets = 4
    patches = math.ceil(PATCHES_PER_SNIPPET * n_snippets)
    values = np.arange(patches, dtype=np.float32)[:, None]
    pooled = pool_audio(values, n_snippets)

    assert pooled.shape == (n_snippets, 1)
    for i in range(n_snippets):
        start = math.ceil(i * PATCHES_PER_SNIPPET)
        stop = min(math.ceil((i + 1) * PATCHES_PER_SNIPPET), patches)
        assert pooled[i, 0] == pytest.approx(values[start:stop, 0].mean())


def test_pool_audio_pads_missing_time_and_empty_input():
    pooled = pool_audio(np.ones((2, 3), dtype=np.float32), 4)   # 2 patches, 4 snippets
    assert pooled.shape == (4, 3)
    assert pooled[0].tolist() == [1.0, 1.0, 1.0]                # both patches sit in snippet 0
    assert pooled[1:].sum() == 0.0                              # the rest is zero-padded
    empty = pool_audio(np.zeros((0, 3), dtype=np.float32), 2)
    assert empty.shape == (2, 3) and empty.sum() == 0.0


def test_pool_audio_grid_is_stride_specific():
    """A 16-frame snippet packs ~0.69 audio patches, not the mirror's ~2.78.

    Using one constant for both grids slides the audio track against the video by up to 4x, so the
    pooling rate must follow the visual stride (this is what an I3D/16-frame head needs, and what
    the live upload path must reproduce).
    """
    from safewatch.data.multimodal import patches_per_snippet

    assert patches_per_snippet(16) == pytest.approx(0.6944, rel=1e-3)
    assert patches_per_snippet(64) == pytest.approx(PATCHES_PER_SNIPPET, rel=1e-3)
    with pytest.raises(ValueError):
        patches_per_snippet(0)

    values = np.arange(40, dtype=np.float32)[:, None]
    mirror = pool_audio(values, 16)                              # default = mirror grid
    official = pool_audio(values, 16, patches_per_snippet(16))   # official grid
    assert not np.array_equal(mirror, official)
    # On the mirror grid 40 patches only reach snippet ~14; on the official grid they fill all 16.
    assert official[-1, 0] > 0.0


# --------------------------------------------------------------------------- #
# fusion model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", FUSION_MODES)
def test_fusion_forward_contract(mode):
    torch.manual_seed(0)
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        n_categories=6, dropout=0.0, fusion_mode=mode, heads=2).eval()
    visual = torch.randn(2, N_SNIPPETS, VISUAL_DIM)
    audio = torch.randn(2, N_SNIPPETS, AUDIO_DIM)
    mask = torch.ones(2, N_SNIPPETS, dtype=torch.bool)
    mask[1, 6:] = False                                         # row 1 is padded after snippet 5

    with torch.no_grad():
        out = model(visual, audio, mask)

    assert out["snippet_logits"].shape == (2, N_SNIPPETS)
    assert out["clip_logits"].shape == (2,)
    assert out["multi_logits"].shape == (2, 6)
    assert torch.isfinite(out["snippet_logits"]).all()
    assert torch.allclose(out["attn"][1, 6:], torch.zeros(4))   # padded snippets: no attention
    assert float(out["attn"][1].sum()) == pytest.approx(1.0)
    # per-modality evidence exists exactly where the explanation panel will expect it
    assert (out["alpha"] is not None) == (mode == "adaptive")
    assert (out["snippet_visual"] is not None) == (mode == "late")
    if out["alpha"] is not None:
        assert out["alpha"].shape == (2, N_SNIPPETS, 1)
        assert float(out["alpha"].min()) >= 0.0
        assert float(out["alpha"].max()) <= 1.0


# --------------------------------------------------------------------------- #
# scoring: input dimension + grid recovery
# --------------------------------------------------------------------------- #
def test_run_config_records_derived_dimensions():
    """config.json / ckpt must carry in_dim and feature_key, not just the feature-set name."""
    cfg_dict = TrainConfig(feature_set="i3d_official").to_dict()
    assert cfg_dict["in_dim"] == OFFICIAL_DIM == 1024
    assert cfg_dict["feature_key"] == "visual"


def test_infer_in_dim_reads_the_weight_shapes():
    mil = MILModel(in_dim=OFFICIAL_DIM, hidden=32, emb=8)
    assert infer_in_dim(mil.state_dict()) == OFFICIAL_DIM
    fusion = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                         fusion_mode="adaptive", heads=2)
    assert infer_in_dim(fusion.state_dict()) == VISUAL_DIM
    assert infer_in_dim({}) is None


def test_load_model_rebuilds_official_checkpoint(tmp_path):
    """Regression: an i3d_official checkpoint was rebuilt as 768-d, so scoring could not run."""
    model = MILModel(in_dim=OFFICIAL_DIM, hidden=64, emb=16)
    config = {**TrainConfig(feature_set="i3d_official").to_dict(), "hidden": 64, "emb": 16}
    ckpt = tmp_path / "mil.pt"
    torch.save({"model": model.state_dict(), "config": config, "epoch": 1}, ckpt)

    loaded, cfg, kind = load_model(ckpt, torch.device("cpu"))
    assert kind == "mil" and loaded.in_dim == OFFICIAL_DIM
    assert resolve_stride(cfg, None) == 16.0        # official release grid
    assert resolve_stride(cfg, 64) == 64.0          # an explicit flag always wins
    with pytest.raises(SystemExit):                 # an unknown grid must not be guessed
        resolve_stride({"feature_set": "c3d_rgb"}, None)


def test_load_model_rebuilds_fusion_checkpoint(tmp_path):
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        fusion_mode="adaptive", heads=2, dropout=0.0)
    ckpt = tmp_path / "fusion.pt"
    torch.save({"model": model.state_dict(),
                "config": {"fusion_mode": "adaptive", "visual_dim": VISUAL_DIM,
                           "audio_dim": AUDIO_DIM, "emb": 8, "hidden": 16, "heads": 2,
                           "dropout": 0.0}, "epoch": 1}, ckpt)

    loaded, cfg, kind = load_model(ckpt, torch.device("cpu"))
    assert kind == "fusion" and model_kind(cfg) == "fusion"
    assert loaded.visual_encoder.net[0].in_channels == VISUAL_DIM
    assert loaded.audio_encoder.net[0].in_channels == AUDIO_DIM
    with torch.no_grad():
        out = loaded(torch.randn(1, N_SNIPPETS, VISUAL_DIM), torch.randn(1, N_SNIPPETS, AUDIO_DIM))
    assert out["snippet_logits"].shape == (1, N_SNIPPETS)
    assert resolve_stride(cfg, None) == 64.0        # fusion ckpts inherit the swin_rgb mirror grid


# --------------------------------------------------------------------------- #
# metric aggregation over a split
# --------------------------------------------------------------------------- #
def _rows_and_scores(labels):
    rows = [{"clip_id": f"c{i}", "labels": lab} for i, lab in enumerate(labels)]
    score_values = [[0.9, 0.1], [0.3, 0.2]]        # clip c1 stays entirely below the 0.5 threshold
    scores = {r["clip_id"]: np.array(score_values[i]) for i, r in enumerate(rows)}
    gt = {r["clip_id"]: np.array([1.0, 0.0], dtype=np.float32) if r["labels"] else
          np.zeros(2, dtype=np.float32) for r in rows}
    gt_segments = {r["clip_id"]: [(0, 0)] if r["labels"] else [] for r in rows}
    multilogits = {r["clip_id"]: np.array([0.9 - 0.1 * i, 0.1, 0.1, 0.1, 0.1, 0.1])
                   for i, r in enumerate(rows)}
    return rows, scores, gt, gt_segments, multilogits


def test_compute_metrics_single_positive_class_gives_macro_ap_one():
    rows, scores, gt, gt_segments, multilogits = _rows_and_scores(["fighting", ""])
    report = compute_metrics(scores, gt, gt_segments, multilogits, rows, 16.0, 0.5, 1, 1)

    assert report["n_clips"] == 2
    assert report["per_class_clip_level_ap"]["fighting"]["n_positive"] == 1
    assert np.isnan(report["per_class_clip_level_ap"]["abuse"]["ap"])   # no positives -> NaN
    assert report["macro_per_class_ap"] == pytest.approx(1.0)           # NaN classes excluded
    assert report["ap"]["global__average_precision"] == pytest.approx(1.0)
    assert report["localization"]["n_pred_segments"] == 1        # clip b stays below threshold


def test_compute_metrics_all_nan_per_class_is_nan_without_warning():
    """A split with no positive clip anywhere must not blow up np.nanmean (all-NaN slice)."""
    rows, scores, gt, gt_segments, multilogits = _rows_and_scores(["", ""])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = compute_metrics(scores, gt, gt_segments, multilogits, rows, 16.0, 0.5, 1, 1)
    assert np.isnan(report["macro_per_class_ap"])


# --------------------------------------------------------------------------- #
# reliability signals: adaptive fusion + robustness study (P8)
# --------------------------------------------------------------------------- #
def test_reliability_channels_layout_and_standardisation():
    """7 channels = [audio quality (5) | visual proxies (2)], standardised on demand."""
    quality = np.full((9, AUDIO_QUALITY_DIM), 2.0, dtype=np.float32)   # 9 patches -> 3 snippets
    visual = np.zeros((3, 4), dtype=np.float32)
    visual[:, 0] = 1.0                                    # ||f||/sqrt(d) = 0.5 per snippet
    channels = reliability_channels(quality, visual, 3)

    assert channels.shape == (3, RELIABILITY_DIM)
    assert channels[:, 0].tolist() == [2.0, 2.0, 2.0]     # pooled audio quality, no stats -> raw
    assert channels[:, AUDIO_QUALITY_DIM].tolist() == [0.5, 0.5, 0.5]   # feature_energy
    stats = {"mean": [2.0] * RELIABILITY_DIM, "std": [1.0] * RELIABILITY_DIM}
    assert reliability_channels(quality, visual, 3, stats)[:, 0].tolist() == [0.0, 0.0, 0.0]


def test_standardise_passes_through_flat_channels():
    """A channel constant in training must not be divided by ~0 (clip_ratio on the mirror)."""
    values = np.array([[0.5, 3.0]], dtype=np.float32)
    stats = {"mean": [0.0, 3.0], "std": [0.0, 1.0]}        # channel 0 has no train variance
    out = standardise(values, stats)
    assert out[0, 0] == pytest.approx(0.5)                 # passed through
    assert out[0, 1] == pytest.approx(0.0)                 # z-scored
    with pytest.raises(ValueError):                        # channel-count mismatch is loud
        standardise(values, {"mean": [0.0], "std": [1.0]})


def test_dataset_reliability_shares_the_audio_window(tmp_path):
    """Reliability is pooled on the same snippet grid as the audio it describes."""
    row = write_clip(tmp_path, clip_id="clip_rel", encode_index=True, quality=True)
    csv_path = write_csv(tmp_path, [row])
    dataset = MultimodalClipDataset(csv_path, tmp_path, modality="both", audio_stats=None,
                                    quality_dir=tmp_path / "quality", quality_stats=None,
                                    max_snippets=None, train=False)
    item = dataset[0]
    q = AUDIO_QUALITY_DIM
    assert item["reliability"].shape == (N_SNIPPETS, RELIABILITY_DIM)
    # every patch of snippet t carries t -> the pooled audio channels must equal t
    assert torch.allclose(item["reliability"][:, 0], item["audio"][:, 0])
    assert torch.allclose(item["reliability"][:, q],
                          torch.arange(N_SNIPPETS, dtype=torch.float32))   # feature_energy
    assert torch.allclose(item["reliability"][1:, q + 1], torch.ones(N_SNIPPETS - 1))


def test_drop_modality_sets_sentinel_and_zeroes_features():
    visual = torch.ones(2, 3, VISUAL_DIM)
    audio = torch.ones(2, 3, AUDIO_DIM)
    reliability = torch.zeros(2, 3, RELIABILITY_DIM)
    v, a, r = drop_modality(visual, audio, reliability, "audio")

    assert torch.all(a == 0) and torch.all(v == 1)                     # only the audio is removed
    assert torch.all(r[..., :AUDIO_QUALITY_DIM] == DEGRADED)           # ... and flagged
    assert torch.all(r[..., AUDIO_QUALITY_DIM:] == 0.0)
    v, a, r = drop_modality(visual, audio, reliability, "visual")
    assert torch.all(v == 0) and torch.all(a == 1)
    v2, a2, r2 = drop_modality(visual, audio, reliability, None)   # modality="none": identity
    assert r2 is reliability and v2 is visual and a2 is audio
    v3, a3, r3 = drop_modality(visual, audio, None, "audio")       # no vector to flag, still zeroed
    assert r3 is None and torch.all(a3 == 0)
    with pytest.raises(ValueError):
        drop_modality(visual, audio, reliability, "flow")


def test_adaptive_gate_reads_the_reliability_channels():
    """The gate must react to the reliability vector, and ignore it when trained without one."""
    torch.manual_seed(0)
    visual = torch.randn(1, N_SNIPPETS, VISUAL_DIM)
    audio = torch.randn(1, N_SNIPPETS, AUDIO_DIM)
    mask = torch.ones(1, N_SNIPPETS, dtype=torch.bool)
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        fusion_mode="adaptive", heads=2, dropout=0.0,
                        n_reliability=RELIABILITY_DIM).eval()
    with torch.no_grad():
        healthy = model(visual, audio, mask,
                        torch.zeros(1, N_SNIPPETS, RELIABILITY_DIM))["alpha"]
        broken = model(visual, audio, mask,
                       torch.full((1, N_SNIPPETS, RELIABILITY_DIM), DEGRADED))["alpha"]
    assert not torch.allclose(healthy, broken)

    blind = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        fusion_mode="adaptive", heads=2, dropout=0.0).eval()   # n_reliability=0
    with torch.no_grad():   # old checkpoints keep working when the dataset now sends a vector
        out = blind(visual, audio, mask, torch.zeros(1, N_SNIPPETS, RELIABILITY_DIM))
    assert out["alpha"].shape == (1, N_SNIPPETS, 1)




def test_clip_normalise_removes_per_clip_gain_and_offset():
    """The calibration fix: a clip-level gain/offset must not change the audio stream's shape."""
    base = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    loud = 3.0 * base + 10.0                                # same content, different level
    assert np.allclose(clip_normalise(base), clip_normalise(loud), atol=1e-5)
    assert np.allclose(clip_normalise(base).mean(axis=0), np.zeros(2), atol=1e-6)


def test_dataset_audio_norm_is_wired_and_toggleable(tmp_path):
    """``audio_norm="clip"`` must change the audio tensor (and nothing else) on the same clip."""
    row = write_clip(tmp_path, clip_id="clip_norm", visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM)
    csv_path = write_csv(tmp_path, [row])
    common = {"modality": "both", "audio_stats": None, "quality_dir": None,
              "max_snippets": None, "train": False}
    plain = MultimodalClipDataset(csv_path, tmp_path, audio_norm="none", **common)[0]
    normed = MultimodalClipDataset(csv_path, tmp_path, audio_norm="clip", **common)[0]

    assert not torch.allclose(plain["audio"], normed["audio"])
    assert torch.allclose(plain["visual"], normed["visual"])
    assert abs(float(normed["audio"].mean())) < 1e-5       # per-channel zero mean
    with pytest.raises(ValueError):
        MultimodalClipDataset(csv_path, tmp_path, audio_norm="global", **common)

def test_snippets_for_seconds_is_grid_independent():
    """2 s of video is 1 snippet on the 64-frame mirror grid but 3 on the official 16-frame grid."""
    assert M.snippets_for_seconds(2.0, 64.0) == 1
    assert M.snippets_for_seconds(2.0, 16.0) == 3
    assert M.snippets_for_seconds(0.1, 64.0, minimum=0) == 0
    assert M.snippets_for_seconds(0.1, 64.0) == 1            # default floor


def test_score_clips_respects_audio_grid_end_to_end(tmp_path):
    """score_clips must forward the checkpoint's ``audio_grid`` to the dataset.

    Regression test for the 2026-09-29 VGGish audit: official-snippet audio files
    (one row per snippet) were pooled as if they were mel patches, so the audio
    track drifted seconds away from the video while shapes stayed plausible.
    """
    from safewatch.data.multimodal import pool_audio

    row = write_clip(tmp_path, clip_id="clip_grid", quality=True)
    csv_path = write_csv(tmp_path, [row])
    n_snippets = int(row["T"])
    n_src = n_snippets * 2  # >2x rows forces pool_audio to average pairs together
    audio = (np.arange(n_src * AUDIO_DIM, dtype=np.float32).reshape(n_src, AUDIO_DIM)
             / (n_src * AUDIO_DIM))
    np.save(tmp_path / "clip_grid.npy", audio)
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        fusion_mode="late", heads=2, dropout=0.0, n_reliability=0)
    ckpt = tmp_path / "fusion_grid.pt"
    torch.save({"model": model.state_dict(), "epoch": 1, "config": {
        "fusion_mode": "late", "visual_dim": VISUAL_DIM, "audio_dim": AUDIO_DIM, "emb": 8,
        "hidden": 16, "heads": 2, "dropout": 0.0, "n_reliability": 0,
        "audio_dir": str(tmp_path), "audio_stats": str(tmp_path / "none.json"),
        "quality_dir": "", "quality_stats": "",
        "audio_grid": "patch", "lists_dir": str(tmp_path)}}, ckpt)

    device = torch.device("cpu")
    dataset = MultimodalClipDataset(csv_path, tmp_path, modality="both", audio_stats=None,
                                    quality_dir="", quality_stats="",
                                    audio_grid="patch", max_snippets=None, train=False)
    expected = pool_audio(audio, n_snippets)
    assert np.allclose(dataset[0]["audio"].numpy(), expected, atol=1e-5)

    (tmp_path / "clip_grid.npy").unlink()
    snippet_audio = np.arange(n_snippets * AUDIO_DIM, dtype=np.float32).reshape(
        n_snippets, AUDIO_DIM) / (n_snippets * AUDIO_DIM)
    np.save(tmp_path / "clip_grid.npy", snippet_audio)
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    cfg["audio_grid"] = "snippet"
    torch.save({"model": model.state_dict(), "epoch": 1, "config": cfg}, ckpt)
    dataset_snippet = MultimodalClipDataset(csv_path, tmp_path, modality="both",
                                            audio_stats=None, quality_dir="",
                                            quality_stats="", audio_grid="snippet",
                                            max_snippets=None, train=False)
    assert np.allclose(dataset_snippet[0]["audio"].numpy(), snippet_audio, atol=1e-5)


    # The scorer itself must honour the checkpoint grid: the same ramp bytes
    # scored as patches vs snippets must feed the model different audio.
    np.save(tmp_path / "clip_grid.npy", audio)  # n_src = 2*T ramp rows
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    cfg["audio_grid"] = "patch"
    torch.save({"model": model.state_dict(), "epoch": 1, "config": cfg}, ckpt)
    scores_patch, _v, _m = score_clips(ckpt, csv_path, device)
    cfg["audio_grid"] = "snippet"
    torch.save({"model": model.state_dict(), "epoch": 1, "config": cfg}, ckpt)
    scores_snippet, _v2, _m2 = score_clips(ckpt, csv_path, device)
    assert not np.allclose(scores_patch["clip_grid"], scores_snippet["clip_grid"])


def test_score_clips_drop_modality_end_to_end(tmp_path):
    """--drop-modality must run through the real scorer and change the score curve."""
    row = write_clip(tmp_path, clip_id="clip_drop", quality=True)
    csv_path = write_csv(tmp_path, [row])
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        fusion_mode="adaptive", heads=2, dropout=0.0,
                        n_reliability=RELIABILITY_DIM)
    ckpt = tmp_path / "fusion_rel.pt"
    torch.save({"model": model.state_dict(), "epoch": 1, "config": {
        "fusion_mode": "adaptive", "visual_dim": VISUAL_DIM, "audio_dim": AUDIO_DIM, "emb": 8,
        "hidden": 16, "heads": 2, "dropout": 0.0, "n_reliability": RELIABILITY_DIM,
        "audio_dir": str(tmp_path), "audio_stats": str(tmp_path / "none.json"),
        "quality_dir": str(tmp_path / "quality"), "quality_stats": str(tmp_path / "none.json"),
        "lists_dir": str(tmp_path)}}, ckpt)

    device = torch.device("cpu")
    full, _valid, _multi = score_clips(ckpt, csv_path, device)
    dropped, _valid, _multi = score_clips(ckpt, csv_path, device, drop="audio")

    assert set(full) == {"clip_drop"}
    assert np.isfinite(dropped["clip_drop"]).all()
    assert not np.allclose(full["clip_drop"], dropped["clip_drop"])


# --------------------------------------------------------------------------- #
# Edge export (P7): the ONNX graph must reproduce the trained model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def export_module():
    """Load ``scripts/export_onnx.py`` (a CLI, not a package module) without importing it as one."""
    import importlib.util
    import sys
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parent.parent / "scripts" / "export_onnx.py"
    spec = importlib.util.spec_from_file_location("safewatch_export_onnx", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_onnx_export_matches_pytorch(tmp_path, export_module):
    """The exported graph must reproduce the trained model - that is what gets deployed.

    Also locks the input contract: early fusion never reads the reliability vector, so the tracer
    prunes it and the graph has three inputs (a hard-coded four-feed call makes ONNX Runtime fail).

    Not asserted here: the legacy TorchScript exporter *can* mutate the model in place. It was
    measured on the real checkpoint (`runs/2026-09-28_p3full_early`: the same model returned logits
    differing by ~8 after `torch.onnx.export`), which is why `scripts/export_onnx.py` captures its
    PyTorch reference *before* exporting; on this tiny model the mutation does not reproduce, so
    asserting it here would be an assertion about a checkpoint, not about the code. Recorded in
    `docs/journal.md`, Session 9c.
    """
    torch.manual_seed(0)
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        fusion_mode="early", heads=2, dropout=0.0).eval()
    inputs = (torch.randn(1, N_SNIPPETS, VISUAL_DIM), torch.randn(1, N_SNIPPETS, AUDIO_DIM),
              torch.ones(1, N_SNIPPETS, dtype=torch.bool), torch.zeros(1, N_SNIPPETS, 1))
    wrapper = export_module.ExportWrapper(model)
    with torch.no_grad():
        reference = wrapper(*inputs)[0].numpy().copy()

    onnx_path = tmp_path / "tiny.onnx"
    torch.onnx.export(wrapper, inputs, str(onnx_path),
                      input_names=["visual", "audio", "mask", "reliability"],
                      output_names=["snippet_logits", "multi_logits"], opset_version=17,
                      dynamo=False)
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = export_module.feeds_for(session, inputs)
    assert set(feeds) == {"visual", "audio", "mask"}     # early fusion never reads reliability
    assert np.abs(reference - session.run(None, feeds)[0]).max() < 1e-4   # the graph is faithful




def test_auto_pos_weight_is_inverse_frequency_and_damped():
    """`abuse` (50/3804 train clips) must dominate the raw weights, and `sqrt` must soften them."""
    from safewatch.data.dataset import CATEGORIES, auto_pos_weight

    rows = [{"labels": "abuse"}] * 50 + [{"labels": "riot"}] * 500
    rows += [{"labels": ""}] * (3804 - 550)                      # clip counts mirror the real split
    raw = auto_pos_weight(rows, "auto")
    damped = auto_pos_weight(rows, "sqrt")
    abuse, riot = CATEGORIES.index("abuse"), CATEGORIES.index("riot")

    assert float(raw[abuse]) == pytest.approx(3754 / 50, rel=1e-6)
    assert float(raw[riot]) == pytest.approx(3304 / 500, rel=1e-6)
    assert float(raw[abuse]) > 5 * float(raw[riot])               # the starvation is real
    assert float(damped[abuse]) == pytest.approx(np.sqrt(float(raw[abuse])), rel=1e-6)
    assert float(damped[abuse]) < float(raw[abuse])
    assert float(auto_pos_weight(rows, "auto", max_weight=10.0)[abuse]) == 10.0   # capped
    with pytest.raises(ValueError):
        auto_pos_weight(rows, "nonsense")


def test_multilabel_loss_weighting_matches_the_reference_implementation():
    """The hand-applied weight must equal ``BCEWithLogitsLoss(pos_weight=...)``; alone, a no-op."""
    from safewatch.data.dataset import N_CATEGORIES
    from safewatch.losses.mil import multilabel_loss

    torch.manual_seed(0)
    logits, target = torch.randn(7, N_CATEGORIES), (torch.rand(7, N_CATEGORIES) < 0.3).float()
    pos_weight = torch.tensor([7.2, 7.9, 9.4, 75.1, 8.0, 9.2])

    weighted = multilabel_loss(logits, target, pos_weight)
    reference = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, target)
    unweighted = multilabel_loss(logits, target)

    assert torch.allclose(weighted, reference, atol=1e-5)
    assert not torch.allclose(weighted, unweighted, atol=1e-5)    # the weight actually bites
    assert torch.allclose(unweighted, torch.nn.BCEWithLogitsLoss()(logits, target), atol=1e-5)


def test_attenuation_endpoints_and_monotonicity():
    """Level 1 must be a no-op, level 0 must collapse onto a constant; in between is monotone."""
    from safewatch.eval.robustness import attenuation

    stream = torch.randn(2, 10, 4)
    mean = torch.zeros(4)
    assert torch.allclose(attenuation(stream, 1.0, mean), stream)
    assert torch.allclose(attenuation(stream, 0.0, mean), torch.zeros_like(stream))
    deltas = [(attenuation(stream, level, mean) - stream).abs().mean().item()
              for level in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert deltas == sorted(deltas)                      # distance to clean grows as level falls
    with pytest.raises(ValueError):
        attenuation(stream, 1.5, mean)


def test_occlusion_uses_seconds_not_snippets():
    """The same duration must zero 2 snippets on the mirror grid and 8 on the official one."""
    from safewatch.eval.robustness import occlude_seconds

    stream = torch.ones(1, 40, 3)
    official = occlude_seconds(stream, 5.333, stride_frames=16.0, seed=0)   # 8 snippets
    mirror = occlude_seconds(stream, 5.333, stride_frames=64.0, seed=0)     # 2 snippets
    assert int((official == 0).all(dim=-1).sum()) == 8
    assert int((mirror == 0).all(dim=-1).sum()) == 2
    assert torch.allclose(occlude_seconds(stream, 5.333, 16.0, seed=0), official)  # deterministic
    assert not torch.allclose(occlude_seconds(stream, 5.333, 16.0, seed=1), official)
    # a window longer than the clip blanks the whole stream instead of raising
    assert (occlude_seconds(stream, 120.0, stride_frames=16.0, seed=0) == 0).all()


def test_drop_snippets_rate_endpoints():
    from safewatch.eval.robustness import drop_snippets

    stream = torch.ones(2, 200, 5)
    assert torch.allclose(drop_snippets(stream, 0.0, seed=0), stream)
    assert (drop_snippets(stream, 1.0, seed=0) == 0).all()
    dropped = drop_snippets(stream, 0.5, seed=0)
    fraction = float((dropped[:, :, 0] == 0).float().mean())
    assert 0.35 < fraction < 0.65                                   # roughly the requested rate
    assert torch.allclose(dropped, drop_snippets(stream, 0.5, seed=0))


def test_add_noise_gets_worse_as_snr_falls():
    from safewatch.eval.robustness import add_noise

    stream = torch.randn(1, 50, 8)
    clean = add_noise(stream, 60.0, None, torch.Generator().manual_seed(0))   # 60 dB: untouched
    mid = add_noise(stream, 10.0, None, torch.Generator().manual_seed(0))
    bad = add_noise(stream, 0.0, None, torch.Generator().manual_seed(0))
    assert (clean - stream).abs().mean() < (mid - stream).abs().mean()
    assert (mid - stream).abs().mean() < (bad - stream).abs().mean()


def test_build_perturbation_skips_missing_streams_and_validates_arguments():
    """A unimodal checkpoint arrives with ``audio=None``: the hook passes it through untouched."""
    from safewatch.eval.robustness import build_perturbation, is_degraded

    cfg = {"feature_set": "i3d_official"}
    hook = build_perturbation("attenuation", 0.0, "audio", cfg, seed=0)
    visual = torch.randn(1, 5, 3)
    out_v, out_a, out_r = hook(visual, None, None)                   # no audio stream to degrade
    assert torch.allclose(out_v, visual) and out_a is None and out_r is None

    audio = torch.randn(1, 5, 4)
    _, out_a, _ = hook(visual, audio, None)
    # level 0 means "no information": in the space the model reads, that is the origin - the same
    # treatment --drop-modality applies, so an attenuation sweep must reproduce that row at level 0
    assert torch.allclose(out_a, torch.zeros_like(audio))

    with pytest.raises(ValueError):
        build_perturbation("nonsense", 0.5, "audio", cfg)
    with pytest.raises(ValueError):
        build_perturbation("attenuation", 0.5, "lidar", cfg)
    assert not is_degraded("attenuation", 1.0)          # clean level -> the sweep installs no hook
    assert is_degraded("attenuation", 0.99)
    assert not is_degraded("occlusion", 0.0)
    assert is_degraded("drop-snippets", 0.1)
    # noise has no clean level: 0 dB is the WORST point, not an untouched one
    assert is_degraded("noise", 0.0)
    assert is_degraded("noise", 40.0)



def test_visual_stream_perturbation_keeps_the_audio_slot():
    """Degrading the visual stream must return it in ITS slot and leave ``audio`` untouched.

    The first version returned the degraded tensor in the audio slot (both branches of a ternary
    were identical). That is silent for equal-width streams and raises a confusing conv1d channel
    error on the official grid (visual 1024 vs audio 128) - what killed the first four occlusion
    sweeps.
    """
    from safewatch.eval.robustness import build_perturbation

    cfg = {"feature_set": "i3d_official"}
    visual = torch.randn(1, 30, 1024)
    audio = torch.randn(1, 30, 128)

    out_v, out_a, _ = build_perturbation("occlusion", 15.0, "visual", cfg, seed=0)(visual, audio,
                                                                                   None)
    assert out_v.shape == visual.shape and out_a.shape == audio.shape
    assert torch.allclose(out_a, audio)                      # audio slot untouched
    assert (out_v == 0).all(dim=-1).sum() > 0                # visual stream actually occluded
    assert not torch.allclose(out_v, visual)

    out_v, out_a, _ = build_perturbation("occlusion", 15.0, "audio", cfg, seed=0)(visual, audio,
                                                                                  None)
    assert torch.allclose(out_v, visual)                     # and the audio path touches audio only
    assert not torch.allclose(out_a, audio)


def test_validate_does_not_track_gradients():
    """``validate`` must not build a graph.

    Its ``@torch.no_grad()`` once ended up decorating the wrong function (a one-line insertion
    between the decorator and its ``def``). The symptom was
    ``RuntimeError: Can't call numpy() on Tensor that requires grad`` at the *first* epoch report -
    i.e. after two minutes of training had already been spent.
    """
    from safewatch.data.dataset import CATEGORIES, N_CATEGORIES
    from safewatch.models.fusion import FusionModel
    from safewatch.train_fusion import validate

    batch = {
        "visual": torch.randn(2, 8, 16),
        "audio": torch.randn(2, 8, 4),
        "mask": torch.ones(2, 8, dtype=torch.bool),
        "binary": torch.tensor([1.0, 0.0]),
        "multi": torch.zeros(2, N_CATEGORIES),
    }

    class _OneBatch:
        def __iter__(self):
            return iter([batch])

    model = FusionModel(visual_dim=16, audio_dim=4, emb=8, hidden=16, n_categories=N_CATEGORIES,
                        dropout=0.0, fusion_mode="late", n_reliability=0)
    report = validate(model, _OneBatch(), torch.device("cpu"))

    assert set(report) >= {"val_ap", "val_pr_auc", "val_multi_ap"}
    assert set(report["val_multi_ap"]) == set(CATEGORIES)



def test_attenuation_level_zero_reproduces_the_drop_modality_contract():
    """Level 0 must zero the stream, exactly like ``--drop-modality`` does before the model.

    The first implementation collapsed onto the *raw* training mean (VGGish mean 0.098) while the
    model reads a normalised stream (mean 0), so "silent" audio actually arrived as a constant DC
    term and the level-0 row could not be reconciled with the drop-modality row.
    """
    from safewatch.data.reliability import drop_modality
    from safewatch.eval.robustness import build_perturbation

    cfg = {"feature_set": "i3d_official", "audio_stats": "data/features/vggish_stats.json"}
    audio = torch.randn(1, 12, 8)
    visual = torch.randn(1, 12, 16)

    _, dropped, _ = drop_modality(visual, audio, None, "audio")
    _, silent, _ = build_perturbation("attenuation", 0.0, "audio", cfg, seed=0)(visual, audio, None)

    assert torch.allclose(dropped, silent)
    assert (silent == 0).all()


def test_noise_reference_is_the_model_space_scale_not_the_raw_release_scale():
    """At 0 dB the noise must be as strong as the signal *as the model sees it*.

    Using the raw VGGish release std (per-dim median 0.222) against a normalised stream (~0.805)
    made the row labelled "0 dB" roughly +10 dB in practice, understating the damage by ~3x.
    """
    from safewatch.eval.robustness import add_noise

    torch.manual_seed(0)
    stream = torch.randn(1, 200, 16)
    noisy = add_noise(stream, 0.0, None, torch.Generator().manual_seed(0))
    ratio = float((noisy - stream).std() / stream.std())
    assert 0.6 < ratio < 1.5                     # noise power ~= signal power

    # an explicit reference is still honoured (for callers who know their space)
    quiet = add_noise(stream, 0.0, torch.full((16,), 0.1), torch.Generator().manual_seed(0))
    assert float((quiet - stream).std() / stream.std()) < 0.3



# --------------------------------------------------------------------------- #
# P4 causal scoring: strided prefix re-forwards, step-held
# --------------------------------------------------------------------------- #
def test_causal_prefix_ends_always_include_the_full_clip():
    """Strided endpoints must tile from the stride and always anchor on the full clip."""
    from safewatch.eval.runner import _causal_prefix_ends

    assert _causal_prefix_ends(10, 8) == [8, 10]
    assert _causal_prefix_ends(5, 8) == [5]     # short clip: single full-clip forward
    assert _causal_prefix_ends(16, 8) == [8, 16]
    assert _causal_prefix_ends(3, 1) == [1, 2, 3]


def test_prefix_tensors_slice_time_but_keep_batch():
    """Every ``(1, T, ...)`` tensor is cut to ``(1, end, ...)``; ``None`` stays ``None``."""
    from safewatch.eval.runner import _prefix_tensors

    tensors = (torch.randn(1, 10, 4), torch.ones(1, 10, dtype=torch.bool))
    sliced = _prefix_tensors(tensors, 4)
    assert [t.shape[1] for t in sliced] == [4, 4]


def test_causal_stride1_final_value_matches_offline_and_holds_between_steps(tmp_path):
    """Causal scoring with stride 1 through the real scorer: same length, finite curve,
    final value == offline score at the last snippet (identical full-clip forward)."""
    row = write_clip(tmp_path, clip_id="clip_causal", quality=True)
    csv_path = write_csv(tmp_path, [row])
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                         fusion_mode="late", heads=2, dropout=0.0, n_reliability=0)
    ckpt = tmp_path / "fusion_causal.pt"
    torch.save({"model": model.state_dict(), "epoch": 1, "config": {
        "fusion_mode": "late", "visual_dim": VISUAL_DIM, "audio_dim": AUDIO_DIM, "emb": 8,
        "hidden": 16, "heads": 2, "dropout": 0.0, "n_reliability": 0,
        "audio_dir": str(tmp_path), "audio_stats": str(tmp_path / "none.json"),
        "quality_dir": "", "quality_stats": "",
        "audio_grid": "patch", "lists_dir": str(tmp_path)}}, ckpt)
    device = torch.device("cpu")
    offline, _v, _m = score_clips(ckpt, csv_path, device)
    causal, _v2, _m2 = score_clips(ckpt, csv_path, device, causal_prefix_stride=1)
    assert causal["clip_causal"].shape == offline["clip_causal"].shape
    assert np.isfinite(causal["clip_causal"]).all()
    assert causal["clip_causal"][-1] == offline["clip_causal"][-1]


def test_intra_video_contrastive_loss_penalizes_flat_anomalies():
    from safewatch.losses.mil import intra_video_contrastive_loss

    # Positive clip with sharp peak vs flat high anomaly
    sharp_logits = torch.tensor([[10.0, 10.0, -10.0, -10.0]])
    flat_logits = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    mask = torch.ones((1, 4), dtype=torch.bool)
    binary = torch.tensor([1.0])

    loss_sharp = intra_video_contrastive_loss(sharp_logits, mask, binary, k=2, margin=0.5)
    loss_flat = intra_video_contrastive_loss(flat_logits, mask, binary, k=2, margin=0.5)

    # Sharp peak has strong top-k vs bottom-k separation -> 0 penalty
    assert loss_sharp.item() == pytest.approx(0.0, abs=1e-4)
    # Flat anomaly has top-k == bottom-k -> maximum margin penalty
    assert loss_flat.item() > 0.45


def test_cross_modal_attention_reliability_modulation():
    from safewatch.models.fusion import CrossModalAttention

    cma = CrossModalAttention(dim=16, heads=2, dropout=0.0)
    cma.eval()

    v = torch.randn(2, 5, 16)
    a = torch.randn(2, 5, 16)
    # Degraded audio stream (low reliability)
    rel_low = torch.full((2, 5, 1), -10.0)

    with torch.no_grad():
        v_out, a_out, w_va, w_av = cma(v, a, reliability=rel_low)

    assert v_out.shape == v.shape
    assert a_out.shape == a.shape
    assert torch.isfinite(v_out).all()


def test_auto_row_weight_boosts_rare_class_clips_only():
    """Exposure remedy for the `abuse` starvation: rare-class rows sampled more, majority untouched."""
    from safewatch.data.dataset import auto_row_weight

    rows = [{"labels": "abuse"}] * 50 + [{"labels": "riot"}] * 500
    rows += [{"labels": ""}] * (3804 - 550)
    uncapped = auto_row_weight(rows, max_weight=float("inf"))

    assert uncapped[:50].min() == pytest.approx(3754 / 50, rel=1e-6)     # abuse rows boosted
    assert uncapped[50:550].min() == pytest.approx(3304 / 500, rel=1e-6) # riot rows lightly boosted
    assert uncapped[550:].min() == 1.0                                  # normals untouched
    assert uncapped[:50].min() > uncapped[50:550].max()                 # abuse rows dominate

    capped = auto_row_weight(rows)                                      # default cap tames the abuse boost
    assert capped[:50].min() == pytest.approx(10.0)
    assert (capped[550:] == 1.0).all()


def test_adaptive_gate_reliability_source_is_signal_driven():
    """gate_source='reliability': a pure measured-signal gate, no embedding input, no memorisation."""
    from safewatch.data.reliability import RELIABILITY_DIM, DEGRADED

    torch.manual_seed(0)
    visual = torch.randn(1, N_SNIPPETS, VISUAL_DIM)
    audio = torch.randn(1, N_SNIPPETS, AUDIO_DIM)
    mask = torch.ones(1, N_SNIPPETS, dtype=torch.bool)

    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        fusion_mode="adaptive", heads=2, dropout=0.0,
                        n_reliability=RELIABILITY_DIM, gate_source="reliability").eval()
    with torch.no_grad():
        healthy = model(visual, audio, mask,
                        torch.zeros(1, N_SNIPPETS, RELIABILITY_DIM))["alpha"]
        video_broken = model(visual, audio, mask,
                             torch.full((1, N_SNIPPETS, RELIABILITY_DIM), DEGRADED))["alpha"]
    assert healthy.shape == (1, N_SNIPPETS, 1)
    assert not torch.allclose(healthy, video_broken)      # reacts to the sentinel
    # the gate sees NOTHING but the 7 channels: changing embeddings must not move it
    other = model(torch.randn(1, N_SNIPPETS, VISUAL_DIM),
                  torch.randn(1, N_SNIPPETS, AUDIO_DIM), mask,
                  torch.zeros(1, N_SNIPPETS, RELIABILITY_DIM))["alpha"]
    assert torch.allclose(healthy, other)
    # and it refuses to run without the channels
    with pytest.raises(ValueError):
        model(visual, audio, mask, None)["alpha"]


def test_fusion_model_default_gate_source_stays_backwards_compatible():
    """Old checkpoints (no gate_source in config) must rebuild with the original embeddings gate."""
    from safewatch.data.reliability import RELIABILITY_DIM

    torch.manual_seed(0)
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        fusion_mode="adaptive", heads=2, dropout=0.0,
                        n_reliability=RELIABILITY_DIM)
    assert model.gate_source == "embeddings"
    gate_in = model.gate.net[0]
    assert gate_in.in_features == 2 * 8 + RELIABILITY_DIM        # the original width, unchanged


def test_load_model_rebuilds_a_reliability_gate_checkpoint(tmp_path):
    """A gate_source=reliability checkpoint must round-trip through the single scoring path."""
    torch.manual_seed(0)
    model = FusionModel(visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, emb=8, hidden=16,
                        n_categories=6, fusion_mode="adaptive", heads=2,
                        n_reliability=RELIABILITY_DIM, gate_source="reliability")
    ckpt = tmp_path / "ckpt_best.pt"
    torch.save({"model": model.state_dict(),
                "config": {"fusion_mode": "adaptive", "visual_dim": VISUAL_DIM,
                           "audio_dim": AUDIO_DIM, "emb": 8, "hidden": 16, "heads": 2,
                           "n_reliability": RELIABILITY_DIM, "gate_source": "reliability"}},
               ckpt)
    rebuilt, cfg, kind = load_model(ckpt, torch.device("cpu"))
    assert kind == "fusion" and cfg["gate_source"] == "reliability"
    visual = torch.randn(1, N_SNIPPETS, VISUAL_DIM)
    audio = torch.randn(1, N_SNIPPETS, AUDIO_DIM)
    mask = torch.ones(1, N_SNIPPETS, dtype=torch.bool)
    rel = torch.zeros(1, N_SNIPPETS, RELIABILITY_DIM)
    with torch.no_grad():
        a = model(visual, audio, mask, rel)["alpha"]
        b = rebuilt(visual, audio, mask, rel)["alpha"]
    assert torch.allclose(a, b)
