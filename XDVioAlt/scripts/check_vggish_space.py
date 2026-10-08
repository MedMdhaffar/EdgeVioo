#!/usr/bin/env python
"""Check which space the official XD-Violence VGGish .npy features live in.

Why this matters
---------------
The AED tagger head (the official VGGish logit layer, 128 -> 1250) is trained on the
*raw* embedding, i.e. the 128-d ReLU output BEFORE the PCA/whitening + 8-bit
quantization that Google applies to the embeddings it *releases* to AudioSet users.
If the .npy files under ``data/features/vggish_snippet`` are post-processed, the head
cannot be applied directly to them.

Two independent checks:

1. **Sign test** - the raw embedding is a ReLU output, hence non-negative; the
   post-PCA embedding is mean-centred whitened and therefore contains negatives.
2. **Correlation test** - recompute the embedding for a handful of snippets with
   ``torchvggish`` (both raw and post-processed) and compare against the released
   rows. Whichever variant tracks the released features is the space we are in.

Usage:
    .venv/bin/python scripts/check_vggish_space.py --clips 3 --snippets 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.edge.audio import decode_audio  # noqa: E402

FEATURES = Path("data/features/vggish_snippet")
LISTS = [Path("data/lists_official") / f"{split}.csv" for split in ("train", "val", "test")]
SNIPPET_S = 16 / 24.0          # official grid: one snippet per 16 frames (~24 fps)
EXAMPLE_S = 0.96               # VGGish input window
HOP_S = 0.01                   # mel frame hop


def find_clip(clip_id: str) -> tuple[dict, Path]:
    for lists in LISTS:
        with lists.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["clip_id"] == clip_id:
                    raw = Path("data/raw/xd-violence/data/video")
                    return row, next(p for p in raw.rglob(f"{clip_id}.mp4"))
    raise FileNotFoundError(clip_id)


def mel_patch(audio: np.ndarray, start_s: float) -> np.ndarray:
    """(96, 64) log-mel patch starting at ``start_s`` seconds (0.96 s window)."""
    from torchvggish import mel_features as mf
    from torchvggish import vggish_params as vp

    log_mel = mf.log_mel_spectrogram(
        audio, audio_sample_rate=vp.SAMPLE_RATE, log_offset=vp.LOG_OFFSET,
        window_length_secs=vp.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vp.STFT_HOP_LENGTH_SECONDS, num_mel_bins=vp.NUM_MEL_BINS,
        lower_edge_hertz=vp.MEL_MIN_HZ, upper_edge_hertz=vp.MEL_MAX_HZ)
    start = int(round(start_s / HOP_S))
    patch = log_mel[start:start + vp.NUM_FRAMES]
    if patch.shape[0] < vp.NUM_FRAMES:      # tail of the clip: zero-pad on the right
        pad = np.zeros((vp.NUM_FRAMES - patch.shape[0], patch.shape[1]), dtype=patch.dtype)
        patch = np.concatenate([patch, pad], axis=0)
    return patch


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument("--snippets", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    names = sorted(p.stem for p in FEATURES.glob("*.npy"))
    picks = list(rng.choice(names, size=args.clips, replace=False))

    print("Loading torchvggish (downloads ~10 MB weights on first run)…")
    from torchvggish.torchvggish import vggish
    model = vggish(postprocess=False).eval()

    verdict_raw, verdict_post = [], []
    for name in picks:
        row, video_path = find_clip(name)
        released = np.load(FEATURES / f"{name}.npy")
        t = int(row["T"])
        assert released.shape[0] == t, f"{name}: {released.shape[0]} rows vs T={t}"

        negatives = int((released < 0).sum())
        print(f"\n== {name}  (T={t})")
        print(f"   released space: min {released.min():.3f}, max {released.max():.3f}, "
              f"negative values: {negatives}/{released.size}")

        audio = decode_audio(video_path)
        if audio.size == 0:
            print("   no audio track - skipping correlation")
            continue
        idx = sorted(rng.integers(0, t, size=min(args.snippets, t)))
        sims_raw, sims_post = [], []
        with torch.no_grad():
            for i in idx:
                patch = mel_patch(audio, i * SNIPPET_S)
                x = torch.from_numpy(patch[None, None].astype(np.float32))
                emb_raw_t = model(x)
                emb_raw = emb_raw_t.numpy().ravel()
                from torchvggish.torchvggish import Postprocessor
                if not hasattr(model, "_pproc"):
                    model._pproc = Postprocessor()
                emb_post = (model._pproc.postprocess(emb_raw_t).reshape(-1)
                            .numpy().astype(np.float32))
                sims_raw.append(cos(emb_raw, released[i]))
                sims_post.append(cos(emb_post, released[i]))
        verdict_raw.append(float(np.mean(sims_raw)))
        verdict_post.append(float(np.mean(sims_post)))
        print(f"   mean cos sim over {len(idx)} snippets: raw {np.mean(sims_raw):.4f} | "
              f"post-PCA {np.mean(sims_post):.4f}")

    print("\n== verdict over", len(picks), "clips")
    print(f"   raw mean cos sim:      {np.mean(verdict_raw):.4f}")
    print(f"   post-PCA mean cos sim: {np.mean(verdict_post):.4f}")
    print("   the released features live in the",
          "RAW (pre-PCA)" if np.mean(verdict_raw) > np.mean(verdict_post)
          else "POST-PCA", "space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
