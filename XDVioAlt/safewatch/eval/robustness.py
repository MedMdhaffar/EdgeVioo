"""Feature-space degradations for the P8 robustness study.

**Read this before quoting a number produced here.** The XD-Violence release this project scores is
*precomputed features*, not media: the visual stream is I3D (official grid) or VideoSwin (mirror)
and the audio stream is either official VGGish or our own 132-d log-mel statistics. Of those, only
the log-mel stream can be regenerated from the source videos (the test mp4s are on disk; measured
cost ~3 s/clip with ffmpeg, so a 4-level SNR sweep over 800 clips is ~2.5 h). VGGish is a released
feature file with no public weights in this repo, and the I3D extractor is absent. Therefore the
sweeps here are **feature-space** perturbations applied at inference time, and must be described
that way:

* :func:`attenuation` - move a stream toward the training mean vector, i.e. progressively delete
  its information while keeping its shape. Interpretable as "audio gain collapses / audio becomes
  uninformative"; level 0 is silence, level 1 is the clean input. It is the feature-space analogue
  of a falling SNR, and its endpoint is what ``--drop-modality`` does.
* :func:`add_noise` - additive Gaussian noise whose sigma is set from the stream's *training* std
  and an SNR in dB. This is the honest way to read "SNR" here: it perturbs the representation,
  not the waveform.
* :func:`occlude_seconds` - zero one contiguous window of the given **wall-clock duration** in one
  modality (camera blocked, microphone muted). Durations are converted with
  :func:`safewatch.eval.metrics.snippets_for_seconds`, because a snippet count means a different
  duration on each grid (the Session 10 trap).
* :func:`drop_snippets` - randomly zero a fraction of a stream's snippets (packet loss / dropped
  frames), independently per clip.

All functions are deterministic given ``seed`` and operate on ``(B, T, D)`` float tensors, editing a
clone; ``None`` streams are skipped so the same callable serves fusion and unimodal checkpoints.
"""
from __future__ import annotations

import torch

from safewatch.eval import metrics as M
from safewatch.eval.runner import STRIDE_FRAMES

KINDS = ("attenuation", "noise", "occlusion", "drop-snippets")
STREAM_NAMES = ("audio", "visual")


def attenuation(stream: torch.Tensor, level: float, mean: torch.Tensor | None) -> torch.Tensor:
    """``mean + level * (stream - mean)``: 1 = untouched, 0 collapses onto ``mean``.

    ``mean`` must be expressed in the space **the model sees**, not in raw feature space. For the
    audio stream that space is the dataset's normalised one (``(x - mean) / std``), so the
    origin is the real "silence": passing the *raw* training mean instead leaves a constant
    DC term behind, which is why the first attenuation sweep had a level-0 row that did not
    match its own ``--drop-modality`` row. ``None`` falls back to the clip's own mean (removes
    temporal information, keeps the level).
    """
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"attenuation level must be in [0, 1], got {level}")
    if mean is None:                       # no training stats: fall back to the clip's own mean
        mean = stream.mean(dim=1, keepdim=True)
    return mean + level * (stream - mean)


def add_noise(stream: torch.Tensor, snr_db: float, std: torch.Tensor | None,
              generator: torch.Generator) -> torch.Tensor:
    """Additive Gaussian noise scaled so the perturbed stream keeps roughly ``snr_db`` SNR.

    ``std`` is the reference standard deviation, and it must be the one **the model sees**: the
    first version fed it the raw VGGish release stats (per-dim median 0.222) while the dataset
    normalises the stream to ~0.805, so the row labelled "0 dB" was really about +10 dB and the
    whole sweep understated the damage by ~3x. ``None`` uses the stream's own per-clip std, which
    is exactly the right reference for a normalised stream: 0 dB then means noise power equals
    signal power.
    """
    reference = std if std is not None else stream.std(dim=1, keepdim=True)
    sigma = reference * (10.0 ** (-snr_db / 20.0))
    noise = torch.randn(stream.shape, generator=generator, device=stream.device, dtype=stream.dtype)
    return stream + sigma * noise


def occlude_seconds(stream: torch.Tensor, seconds: float, stride_frames: float,
                    seed: int, fps: float = 24.0) -> torch.Tensor:
    """Zero one contiguous window of ``seconds`` per clip, at a seeded random offset."""
    n_snippets = stream.shape[1]
    length = M.snippets_for_seconds(seconds, stride_frames, fps, minimum=1)
    if length >= n_snippets:
        return torch.zeros_like(stream)
    generator = torch.Generator().manual_seed(seed)
    offset = int(torch.randint(0, n_snippets - length + 1, (1,), generator=generator).item())
    out = stream.clone()
    out[:, offset:offset + length, :] = 0.0
    return out


def drop_snippets(stream: torch.Tensor, rate: float, seed: int) -> torch.Tensor:
    """Zero each snippet independently with probability ``rate``."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"drop rate must be in [0, 1], got {rate}")
    generator = torch.Generator().manual_seed(seed)
    keep = torch.rand(stream.shape[1], generator=generator) >= rate
    return stream * keep.to(stream.device).view(1, -1, 1)


def is_degraded(kind: str, level: float) -> bool:
    """True when this level actually perturbs the stream (flags the label, never the score).

    The three families do not share a convention, and conflating them silently skips the worst row:

    * ``attenuation`` - 1.0 is the clean input, 0.0 is silence, so only 1.0 is "not degraded";
    * ``noise`` - **any** finite SNR adds noise, and a *lower* SNR is worse, so 0 dB is the most
      degraded point, not a clean one. (Treating 0 as clean made the first noise sweep silently
      score its worst row with no perturbation at all and report it as the clean reference.)
    * ``occlusion`` / ``drop-snippets`` - 0 means "nothing removed".
    """
    if kind == "attenuation":
        return level != 1.0
    if kind == "noise":
        return True
    return level > 0.0


def build_perturbation(kind: str, level: float, stream: str, cfg: dict, seed: int = 0,
                       flag_reliability: bool = False):
    """Return the ``(visual, audio, reliability) -> (...)`` hook for :func:`runner.score_clips`.

    ``stream`` selects what is degraded (``"audio"`` / ``"visual"``). ``flag_reliability`` also
    marks the degraded stream in the reliability vector, which only the ``adaptive`` gate can read -
    so "gate blind vs gate told" is a meaningful comparison for that variant alone.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown degradation {kind!r}; expected one of {KINDS}")
    if stream not in STREAM_NAMES:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAM_NAMES}")
    stride = STRIDE_FRAMES.get(str(cfg.get("feature_set")), 64.0)
    # Reference point of "no information" is the ORIGIN of the space the model reads. Both streams
    # of the official grid are fed normalised by the dataset (audio via audio_stats), and zeroing a
    # feature stream is also exactly what --drop-modality does - so level 0 of `attenuation` must
    # reproduce that row. No raw-space (mean, std) is passed down: mixing the two spaces is what
    # mislabelled the first noise sweep by ~10 dB.
    origin = torch.zeros(1)
    generator = torch.Generator().manual_seed(seed)

    def hook(visual, audio, reliability):
        target = audio if stream == "audio" else visual
        if target is None:                       # unimodal checkpoint that lacks this stream
            return visual, audio, reliability
        if kind == "attenuation":
            out = attenuation(target, level, origin.to(target.device))
        elif kind == "noise":
            out = add_noise(target, level, None, generator)
        elif kind == "occlusion":
            out = occlude_seconds(target, level, stride, seed)
        else:
            out = drop_snippets(target, level, seed)
        if flag_reliability and reliability is not None:
            from safewatch.data.reliability import degrade_reliability

            reliability = degrade_reliability(reliability, stream)
        # Put the degraded stream back in ITS OWN slot. Returning it in the wrong slot is silent for
        # same-width streams and raises a confusing conv1d channel error for the others (visual 1024
        # vs audio 128), which is how this was caught.
        return (visual, out, reliability) if stream == "audio" else (out, audio, reliability)

    return hook
