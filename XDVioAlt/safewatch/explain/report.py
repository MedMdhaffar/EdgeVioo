"""Alert cards: turn a scored clip into the explanation the operator reads.

The subject lists seven things every alert must show; each one is a field here, and every field is
derived from something already computed rather than invented:

| subject requirement | field | where it comes from |
|---|---|---|
| detected category | ``category`` (+ all 6) | the multi-label head, calibrated |
| temporal segment | ``segments`` | snippet curve + threshold/merge rule **in s** |
| significant images/zones | ``visual_evidence`` + ``media`` | attention + frames + motion zones |
| sound events | ``audio_evidence.acoustic_events`` | YAMNet named tags + channels |
| contribution of each modality | ``modality_contribution`` | per-modality curves or drop-one |
| confidence | ``confidence`` | temperature-scaled probability + its ECE evidence |
| concise textual explanation | ``text`` | template over the fields above (fr/en) |

Three honesty rules encoded in the output:
* the model is **weakly supervised**: it knows a clip contains an anomaly and roughly where, so an
  alert says "segment suspect", never "event confirmed". Human verification is always required.
* the quality channels are signal statistics, not a classifier: they are described ("broadband",
  "saturated"), never over-interpreted.
* the *named* sound events come from a **frozen, out-of-domain** tagger (YAMNet, AudioSet-YouTube
  521 classes, ``safewatch.edge.aed``): it can name "Explosion"/"Gunshot, gunfire" with a measured
  probability, but it was never trained on XD-Violence, so the card presents it as corroboration
  (an "acoustic signature"), never as the reason for the alert.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safewatch.data.dataset import CATEGORIES
from safewatch.eval import metrics as M

UNKNOWN = "unknown"


@dataclass
class AlertConfig:
    """Thresholds and display knobs - all in **seconds** so the card is grid-independent."""

    threshold: float = 0.5
    min_seconds: float = 5.0
    max_gap_seconds: float = 2.5
    top_k_evidence: int = 5
    language: str = "fr"
    category_adaptive: bool = False


@dataclass
class ClipContext:
    """Everything needed to explain one clip: the scores, the evidence, and where they came from."""

    clip_id: str
    stride_frames: float
    fps: float
    n_snippets: int
    snippet_probability: np.ndarray
    attention: np.ndarray | None = None
    alpha: np.ndarray | None = None
    snippet_visual: np.ndarray | None = None
    snippet_audio: np.ndarray | None = None
    category_probability: dict = field(default_factory=dict)
    reliability: np.ndarray | None = None
    duration_s: float | None = None
    video_path: str | None = None
    quality: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    #: Per-snippet YAMNet tag probabilities, shape ``(T, 521)`` float - None when the clip has no
    #: acoustic-event tags (no audio track, or ``extract_aed_tags.py`` was never run for it).
    aed_tags: np.ndarray | None = None

    @property
    def stride_seconds(self) -> float:
        return self.stride_frames / self.fps


def segment_bounds(context: ClipContext, config: AlertConfig) -> list[tuple[int, int]]:
    """Alert segments as inclusive snippet ranges, with the rule expressed in seconds."""
    top_cat = None
    if getattr(config, "category_adaptive", False) and context.category_probability:
        ranked = sorted(context.category_probability.items(), key=lambda kv: kv[1], reverse=True)
        if ranked and ranked[0][1] >= 0.15:
            top_cat = ranked[0][0]

    return M.category_adaptive_segments(
        context.snippet_probability,
        category=top_cat if getattr(config, "category_adaptive", False) else None,
        stride_frames=context.stride_frames,
        fps=context.fps,
        default_threshold=config.threshold,
        default_min_seconds=config.min_seconds,
        default_max_gap_seconds=config.max_gap_seconds,
    )


def _snippet_to_seconds(context: ClipContext, index: int) -> float:
    return float(index * context.stride_seconds)


def describe_segments(context: ClipContext, segments, threshold: float) -> list[dict]:
    out = []
    for start, end in segments:
        window = context.snippet_probability[start:end + 1]
        out.append({
            "start_s": round(_snippet_to_seconds(context, start), 2),
            "end_s": round(_snippet_to_seconds(context, end + 1), 2),
            "duration_s": round((end - start + 1) * context.stride_seconds, 2),
            "n_snippets": int(end - start + 1),
            "mean_probability": round(float(window.mean()), 4),
            "peak_probability": round(float(window.max()), 4),
            "n_snippets_above_threshold": int((window >= threshold).sum()),
        })
    return out


def visual_evidence(context: ClipContext, top_k: int) -> dict:
    """The moments the model scored highest, ranked by importance (ties: attention, then time).

    The list is ordered by *importance*, not by time: an operator triaging an alert reads "the
    snippets that drove the decision" first and gets their place on the timeline from ``second``.
    ``lexsort`` keeps the tie-break deterministic, which matters because XD-Violence scores
    saturate: a plain ``argsort`` reverses tied snippets, so equal-score moments would swap rank
    between runs.
    """
    probability = np.asarray(context.snippet_probability, dtype=float)
    attention = context.attention
    usable_attention = (attention is not None and probability.size > 0
                        and attention.size == probability.size)
    secondary = (np.asarray(attention, dtype=float) if usable_attention
                 else np.zeros_like(probability))
    order = np.lexsort((np.arange(probability.size), -secondary, -probability))[:top_k]
    evidence: dict = {
        "criterion": "ranked by snippet score"
        + (", ties broken by temporal attention" if usable_attention else ""),
        "top_snippets": [
            {"snippet": int(i), "second": round(_snippet_to_seconds(context, int(i)), 2),
             "probability": round(float(probability[int(i)]), 4),
             **({"attention": round(float(context.attention[int(i)]), 5)}
                if usable_attention else {})}
            for i in order
        ],
    }
    if usable_attention:
        peak = int(np.argmax(context.attention))
        evidence["peak_attention"] = {"snippet": peak,
                                      "second": round(_snippet_to_seconds(context, peak), 2),
                                      "attention": round(float(context.attention[peak]), 5)}
        # attention entropy: a flat curve means "no moment stands out" - worth saying out loud
        weights = np.clip(context.attention, 0, None)
        if weights.sum() > 0:
            normalised = weights / weights.sum()
            entropy = float(-(normalised * np.log(np.clip(normalised, 1e-12, None))).sum())
            evidence["attention_entropy"] = round(entropy, 4)
            evidence["attention_entropy_normalised"] = round(
                entropy / np.log(max(2, normalised.size)), 4)
    return evidence



def audio_events(context: ClipContext, segments, top_k: int = 3) -> dict:
    """Named sound events per segment, from the frozen YAMNet tagger (``safewatch.edge.aed``).

    Semantics, deliberately:

    * ``events`` - the incident-relevant classes (the curated ``aed.INCIDENT_TAG_SET``), ranked by
      their **peak** probability inside the segment, each with the time of its peak. A segment is
      mostly quiet speech with one 2 s explosion: the operator must see "Explosion (0.73 @ 00:19)",
      not "Speech (0.18)" - the mean over the whole segment would bury the event.
    * ``incident_energy`` - mean over snippets of the mean probability of the curated classes:
      the fraction of the segment that *sounds incident-like* (0.32 = "32 % of the segment"). This
      is the snippet-level indicator the extractor audits against the ground truth.

    Everything is labelled corroboration - the tagger was trained on AudioSet-YouTube, not on
    XD-Violence, and the card never presents it as the reason for the alert.
    """
    if context.aed_tags is None or np.asarray(context.aed_tags).size == 0:
        return {"available": False,
                "note": "no acoustic-event tags for this clip (run scripts/extract_aed_tags.py, "
                        "or the clip has no audio track)"}
    from safewatch.edge import aed  # function-level: keeps this module importable without torch
    tags = np.asarray(context.aed_tags, dtype=np.float64)
    try:
        names = aed.tag_names()
        idx, _ = aed.incident_tag_indices(names)
    except (FileNotFoundError, ValueError) as exc:
        return {"available": False, "note": f"class map unavailable: {exc}"}

    energy = tags[:, idx].mean(axis=1)
    out: dict = {"available": True,
                 "tagger": "YAMNet (AudioSet-YouTube, 521 classes, frozen)",
                 "corroboration_only": True,
                 "incident_energy": {
                     "mean": round(float(energy.mean()), 4),
                     "peak": round(float(energy.max()), 4),
                     "peak_second": round(float(np.argmax(energy) * context.stride_seconds), 2),
                     "classes": [names[int(i)] for i in idx],
                 }}
    if segments:
        per_segment = []
        for start, end in segments:
            window = tags[start:end + 1]                      # (n_seg, 521)
            seg_energy = energy[start:end + 1]
            # rank the curated classes by their PEAK inside the segment; a class that never
            # exceeds a negligible probability is not worth naming
            ranked = sorted(
                ((int(j), float(window[:, int(j)].max())) for j in idx),
                key=lambda pair: -pair[1])[:top_k]
            events = []
            for class_index, peak in ranked:
                if peak < 0.05:
                    continue
                peak_snippet = int(np.argmax(window[:, class_index]))
                events.append({
                    "label": names[class_index],
                    "probability": round(peak, 4),
                    "at_second": round(_snippet_to_seconds(context, start + peak_snippet), 2),
                })
            per_segment.append({
                "start_s": round(_snippet_to_seconds(context, start), 2),
                "end_s": round(_snippet_to_seconds(context, end + 1), 2),
                "events": events,
                "incident_energy": round(float(seg_energy.mean()), 4),
            })
        out["segments"] = per_segment
    return out


def audio_evidence(context: ClipContext, segments, top_k: int = 3) -> dict:
    """What the audio channels show inside the segments - described, not over-claimed.

    Two layers, both measured: the quality channels (``[rms, rms_db, flatness, zcr, clip_ratio]``
    per snippet, see ``safewatch/data/reliability.py``) and - when available - the *named* events
    of the frozen YAMNet tagger (``audio_events``). ``rms_db`` above the clip's own median means
    "louder than this clip's background"; ``flatness`` high means noise-like rather than tonal;
    ``clip_ratio`` above zero means the channel saturated.
    """
    info: dict = {"channels": ["rms", "rms_db", "flatness", "zcr", "clip_ratio"],
                  "available": context.reliability is not None}
    info["acoustic_events"] = audio_events(context, segments, top_k=top_k)
    #: legacy signature strings (impact_transient, ...) measured on the raw waveform - kept as a
    #: separate key so the channel-derived phrases and the named YAMNet events never mix
    info["signature_events"] = list(context.meta.get("acoustic_events") or [])
    if context.reliability is None:
        info["note"] = "no reliability channels for this run (trained before Session 9)"
    else:
        quality = np.asarray(context.reliability)
        inside = np.zeros(quality.shape[0], dtype=bool)
        for start, end in segments:
            inside[start:end + 1] = True
        if not inside.any():                  # no segment: describe the loudest moments instead
            inside[np.argsort(quality[:, 1])[::-1][:top_k]] = True

        rms_db = quality[:, 1]
        baseline = float(np.median(rms_db))
        segment_values = rms_db[inside]
        info["segment_vs_background_db"] = round(float(segment_values.mean() - baseline), 2)
        info["rms_db"] = {"segment_mean": round(float(segment_values.mean()), 2),
                          "clip_median": round(baseline, 2),
                          "segment_peak": round(float(segment_values.max()), 2)}
        info["spectral_flatness"] = round(float(quality[inside, 2].mean()), 4)
        info["clipping_ratio_max"] = round(float(quality[:, 4].max()), 5)
    info["descriptions"] = describe_audio_phrases(info)
    return info


def _named_event_phrases(info: dict) -> list[str]:
    """Phrases from the named YAMNet events, strongest segment first (corroboration only)."""
    acoustic = info.get("acoustic_events") or {}
    if not acoustic.get("available"):
        return []
    phrases: list[str] = []
    segments = acoustic.get("segments") or []
    if segments:
        ranked = sorted(segments, key=lambda s: -s["incident_energy"])
        top = ranked[0]
        if top["events"]:
            names = ", ".join(
                f"{e['label']} ({e['probability']:.2f} @ {e['at_second']:.1f}s)"
                for e in top["events"][:3])
            phrases.append(
                f"evenements sonores nommes (YAMNet) dans le segment "
                f"{top['start_s']:.1f}-{top['end_s']:.1f}s: {names}; signature d'incident sur "
                f"{top['incident_energy'] * 100:.0f} % du segment"
                f" / named sound events (YAMNet) in the "
                f"{top['start_s']:.1f}-{top['end_s']:.1f}s segment: {names}; incident signature on "
                f"{top['incident_energy'] * 100:.0f} % of it")
        else:
            phrases.append("aucun evenement sonore d'incident nomme dans le segment (YAMNet) / "
                           "no incident sound event named in the segment (YAMNet)")
    else:
        peak = acoustic.get("incident_energy", {})
        if peak.get("peak", 0.0) > 0.05:
            phrases.append(
                f"signature acoustique d'incident a {peak.get('peak_second')}s "
                f"(p={peak.get('peak'):.2f}, YAMNet) / incident acoustic signature at "
                f"{peak.get('peak_second')}s (p={peak.get('peak'):.2f}, YAMNet)")
    return phrases


def describe_audio_phrases(info: dict) -> list[str]:
    """Plain-language phrases derived from the measured channels (fr + en kept side by side).

    Named YAMNet events come first when available - they are the subject's
    "evenements sonores identifies" - followed by the channel-derived cues.
    """
    phrases: list[str] = _named_event_phrases(info)
    events = info.get("signature_events") or []
    if "impact_transient" in events:
        phrases.append(
            "choc/impact acoustique soudain (collision) / sudden acoustic impact shock"
        )
    if "screech_harmonic" in events:
        phrases.append(
            "crissement aigu ou alarme/sirene (freinage) / tire screech or siren"
        )
    if "friction_shatter" in events:
        phrases.append(
            "frottement de surface ou bris de verre/metal / glass/metal friction"
        )

    delta = info.get("segment_vs_background_db")
    if delta is not None and delta >= 6.0:
        phrases.append(f"niveau sonore nettement au-dessus du fond (+{delta:.1f} dB) / "
                       "sound level well above the clip's background")
    elif delta is not None and delta <= -6.0:
        phrases.append(f"segment plus silencieux que le fond ({delta:.1f} dB) / quieter than "
                       "the clip's background")
    flatness = info.get("spectral_flatness")
    if flatness is not None and flatness > 0.35 and "broadband_blast" not in events:
        phrases.append("spectre proche d'un bruit large bande (impact, souffle) / broadband, "
                       "noise-like spectrum")
    elif flatness is not None and flatness < 0.05 and "screech_harmonic" not in events:
        phrases.append("spectre tonal (sirene, moteur, voix tenue) / tonal spectrum")
    clipping = info.get("clipping_ratio_max")
    if clipping is not None and clipping > 0.01:
        phrases.append(f"signal sature sur {clipping * 100:.1f} % des echantillons / channel "
                       "saturation")
    if not phrases:
        phrases.append("aucun indice sonore marquant / no distinctive audio cue")
    return phrases


def modality_contribution(context: ClipContext) -> dict:
    """How much each modality carries, from the model's own outputs and from a drop-one comparison.

    ``snippet_visual`` / ``snippet_audio`` exist for late fusion; ``alpha`` for the adaptive gate.
    When neither exists (early/cross fusion) the card falls back to the drop-one deltas that
    ``ClipContext.meta["drop_one"]`` carries, computed by the caller with the same contract the
    robustness study uses (``data.reliability.drop_modality``). Reporting both is intentional: the
    per-modality curves show *where* a modality fires, the deltas show *how much* the decision
    changed without it.
    """
    out: dict = {"available": False}
    if context.snippet_audio is not None and context.snippet_visual is not None:
        audio = np.asarray(context.snippet_audio)
        visual = np.asarray(context.snippet_visual)
        out.update({
            "available": True,
            "type": "per-modality score curves (late fusion)",
            "mean_probability": {"visual": round(float(visual.mean()), 4),
                                 "audio": round(float(audio.mean()), 4)},
            "peak_probability": {"visual": round(float(visual.max()), 4),
                                 "audio": round(float(audio.max()), 4)},
        })
    if context.alpha is not None:
        alpha = np.asarray(context.alpha).reshape(-1)
        previous = out.get("type")
        out.update({
            "available": True,
            "type": f"{previous} + adaptive gate" if previous else "adaptive gate",
            "gate_mean": round(float(alpha.mean()), 4),
            "gate_range": [round(float(alpha.min()), 4), round(float(alpha.max()), 4)],
            "gate_note": "1 = trust video, 0 = trust audio (per snippet)",
        })
    if context.meta.get("drop_one"):
        out["available"] = True
        drop_meta = context.meta["drop_one"]
        out["drop_one"] = drop_meta
        previous = out.get("type")
        out["type"] = f"{previous} + score without each modality" if previous else \
            "score without each modality"
        if "mean_probability" not in out and isinstance(drop_meta, dict):
            dv = float(drop_meta.get("visual", {}).get("delta", 0.0))
            da = float(drop_meta.get("audio", {}).get("delta", 0.0))
            pv = max(0.0, dv)
            pa = max(0.0, da)
            total = pv + pa
            if total > 1e-6:
                out["mean_probability"] = {
                    "visual": round(pv / total, 2),
                    "audio": round(pa / total, 2),
                }
            else:
                out["mean_probability"] = {"visual": 0.50, "audio": 0.50}
    if not out["available"]:
        out["note"] = ("no per-modality output for this fusion mode and no drop-one comparison was "
                       "requested; run with --drop-one to measure it")
    return out


def confidence_block(context: ClipContext, config: AlertConfig, calibration=None,
                     k: int = 5) -> dict:
    """Calibrated probabilities for the clip and for the strongest segment.

    ``clip_probability`` uses the top-k mean of the snippet curve (the training objective's own
    rule); ``alert_probability`` uses the mean over the strongest segment, i.e. what the operator is
    being asked to look at. Both are mapped through the fitted temperature when one is available -
    and when it is not, the block says ``calibrated: false`` instead of printing a raw sigmoid as if
    it were a probability.
    """
    probability = np.asarray(context.snippet_probability, dtype=np.float64)
    k_eff = max(1, min(k, probability.size))
    clip_raw = float(np.sort(probability)[-k_eff:].mean())
    segments = segment_bounds(context, config)
    if segments:
        strongest = max(segments, key=lambda s: float(probability[s[0]:s[1] + 1].mean()))
        alert_raw = float(probability[strongest[0]:strongest[1] + 1].mean())
    else:
        alert_raw = clip_raw

    block = {
        "raw_clip_probability": round(clip_raw, 4),
        "raw_alert_probability": round(alert_raw, 4),
        "top_k": k_eff,
        "calibrated": calibration is not None and not calibration.is_identity,
    }
    if calibration is not None:
        block["temperature"] = round(calibration.temperature, 4)
        block["clip_probability"] = round(float(calibration.probability(np.array([clip_raw]))[0]),
                                          4)
        block["alert_probability"] = round(
            float(calibration.probability(np.array([alert_raw]))[0]), 4)
        metrics = calibration.metrics or {}
        if "ece_after" in metrics:
            block["evidence"] = {
                "ece_before": round(float(metrics.get("ece_before", float("nan"))), 4),
                "ece_after": round(float(metrics.get("ece_after", float("nan"))), 4),
                "fitted_on": calibration.fitted_on.get("split", UNKNOWN),
                "n_samples": calibration.fitted_on.get("n_samples"),
                "note": "temperature fitted on the validation split; applied to snippet scores as "
                        "well (shared-scale assumption, see safewatch/eval/calibration.py)",
            }
    else:
        block["clip_probability"] = round(clip_raw, 4)
        block["alert_probability"] = round(alert_raw, 4)
        block["note"] = ("no calibration.json for this run: the numbers are raw sigmoid outputs "
                         "and must not be read as probabilities - run "
                         "scripts/fit_calibration.py")
    return block


def category_block(context: ClipContext, top_k: int = 3) -> dict:
    """Detected category (top-1) plus the ranked alternatives.

    The temperature is **not** applied here: it was fitted on the anomaly score, and pretending it
    calibrates a 6-class head would be false precision. The values are raw sigmoid outputs and the
    note says so.
    """
    if not context.category_probability:
        return {"available": False,
                "note": "this checkpoint has no category head output for the clip"}
    items = sorted(context.category_probability.items(), key=lambda kv: kv[1], reverse=True)
    block = {
        "available": True,
        "label": items[0][0],
        "probability": round(float(items[0][1]), 4),
        "ranked": [{"category": name, "probability": round(float(value), 4)}
                   for name, value in items[:top_k]],
        "all": {name: round(float(value), 4) for name, value in items},
        "note": "raw sigmoid outputs of the multi-label head (uncalibrated), trained on weak clip "
                "labels; 'abuse' has only 8 test clips so its probability is the least reliable",
    }
    return block



TEXT_TEMPLATES = {
    "fr": {
        "header": "Incident detecte entre {start} et {end}.",
        "no_segment": "Aucun segment au-dessus du seuil {threshold} : clip considere comme normal.",
        "category": "Categorie probable : {label} ({probability:.0%}).",
        "visual": "Indices visuels : {detail}.",
        "audio": "Indices audio : {detail}.",
        "contribution": "Contribution des modalites : {detail}.",
        "confidence": "Confiance globale : {probability:.0%}{calibrated}.",
        "calibrated_suffix": " (calibree, temperature {temperature:g})",
        "uncalibrated_suffix": " (non calibree - lire comme un score, pas une probabilite)",
        "verification": "Verification humaine recommandee avant toute action.",
        "weak_supervision": ("Modele faiblement supervise : il localise une zone suspecte, il ne "
                             "confirme pas l'evenement."),
    },
    "en": {
        "header": "Incident detected between {start} and {end}.",
        "no_segment": "No segment above threshold {threshold}: clip treated as normal.",
        "category": "Likely category: {label} ({probability:.0%}).",
        "visual": "Visual evidence: {detail}.",
        "audio": "Audio evidence: {detail}.",
        "contribution": "Modality contribution: {detail}.",
        "confidence": "Overall confidence: {probability:.0%}{calibrated}.",
        "calibrated_suffix": " (calibrated, temperature {temperature:g})",
        "uncalibrated_suffix": " (uncalibrated - a score, not a probability)",
        "verification": "Human verification recommended before any sensitive action.",
        "weak_supervision": ("Weakly supervised model: it localises a suspicious window, it does "
                             "not confirm the event."),
    },
}


def _clock(seconds: float) -> str:
    """mm:ss (the subject's alert example uses ``01:24``)."""
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{rest:02d}"


def render_text(alert: dict, language: str = "fr") -> str:
    """Concise operator text; every sentence is filled from a measured field of the alert."""
    template = TEXT_TEMPLATES.get(language, TEXT_TEMPLATES["fr"])
    lines: list[str] = []
    segments = alert.get("segments", [])
    if not segments:
        lines.append(template["no_segment"].format(threshold=alert["config"]["threshold"]))
    else:
        strongest = max(segments, key=lambda s: s["peak_probability"])
        lines.append(template["header"].format(start=_clock(strongest["start_s"]),
                                              end=_clock(strongest["end_s"])))
    category = alert.get("category", {})
    if category.get("available"):
        lines.append(template["category"].format(label=category["label"],
                                                 probability=category["probability"]))
    visual = alert.get("visual_evidence", {}).get("top_snippets", [])
    if visual:
        top = visual[0]
        lines.append(template["visual"].format(
            detail=f"snippet le plus anormal a {_clock(top['second'])} (p={top['probability']:.2f})"
            if language == "fr" else
            f"most anomalous snippet at {_clock(top['second'])} (p={top['probability']:.2f})"))
    audio = alert.get("audio_evidence", {})
    if audio.get("descriptions"):
        lines.append(template["audio"].format(detail="; ".join(audio["descriptions"])))
    contribution = alert.get("modality_contribution", {})
    detail = contribution.get("type")
    if contribution.get("mean_probability"):
        means = contribution["mean_probability"]
        detail = (f"{detail} - visuel {means['visual']:.2f} vs audio {means['audio']:.2f}"
                  if language == "fr" else
                  f"{detail} - visual {means['visual']:.2f} vs audio {means['audio']:.2f}")
    elif contribution.get("drop_one"):
        drops = contribution["drop_one"]
        detail = ", ".join(f"sans {k}: {v['global_ap']:.3f}" if language == "fr"
                           else f"without {k}: {v['global_ap']:.3f}" for k, v in drops.items())
    if detail:
        lines.append(template["contribution"].format(detail=detail))
    confidence = alert.get("confidence", {})
    probability = confidence.get("alert_probability")
    if probability is not None:
        if confidence.get("calibrated"):
            suffix = template["calibrated_suffix"].format(temperature=confidence["temperature"])
        else:
            suffix = template["uncalibrated_suffix"]
        lines.append(template["confidence"].format(probability=probability, calibrated=suffix))
    lines.append(template["verification"])
    lines.append(template["weak_supervision"])
    return " ".join(lines)


def build_alert(context: ClipContext, config: AlertConfig | None = None, calibration=None,
                media: dict | None = None) -> dict:
    """Assemble the full alert card (JSON-serialisable) for one clip.

    ``media`` carries paths produced elsewhere (key frames, spectrogram) so this function stays
    pure - it can then be unit-tested without ffmpeg, OpenCV or matplotlib, and the UI can call it
    in-process.
    """
    config = config or AlertConfig()
    segments = segment_bounds(context, config)
    alert = {
        "clip_id": context.clip_id,
        "video": context.video_path,
        "duration_s": context.duration_s,
        "grid": {"stride_frames": context.stride_frames, "fps": context.fps,
                 "stride_s": round(context.stride_seconds, 3), "n_snippets": context.n_snippets},
        "config": {"threshold": config.threshold, "min_seconds": config.min_seconds,
                   "max_gap_seconds": config.max_gap_seconds,
                   "language": config.language},
        "category": category_block(context),
        "segments": describe_segments(context, segments, config.threshold),
        "visual_evidence": visual_evidence(context, config.top_k_evidence),
        "audio_evidence": audio_evidence(context, segments),
        "modality_contribution": modality_contribution(context),
        "confidence": confidence_block(context, config, calibration),
        "media": media or {},
        "requires_human_verification": True,
        "caveats": [
            "weak supervision: clip-level labels during training, frame-level only for evaluation",
            f"temporal resolution is the snippet grid ({context.stride_seconds:.2f} s here)",
            "category probabilities come from an uncalibrated multi-label head",
            ("named sound events come from a frozen out-of-domain tagger (YAMNet, "
             "AudioSet-YouTube): corroboration, not the reason for the alert"
             if context.aed_tags is not None
             else "no acoustic-event tagger output for this clip (signal statistics only)"),
        ],
        "provenance": context.meta.get("provenance", {}),
    }
    alert["text"] = render_text(alert, config.language)
    return alert


def summarise(alert: dict, max_segments: int = 3) -> str:
    """One-line-per-item console summary (used by the CLI and by scripts/status-style checks)."""
    lines = [f"{alert['clip_id']}"]
    if alert["segments"]:
        for segment in alert["segments"][:max_segments]:
            window = (f"  segment {_clock(segment['start_s'])}-{_clock(segment['end_s'])} "
                      f"({segment['duration_s']:.1f} s)")
            quality = (f" p_mean={segment['mean_probability']:.2f}"
                       f" p_peak={segment['peak_probability']:.2f}")
            lines.append(window + quality)
    else:
        lines.append("  no segment above threshold")
    category = alert.get("category", {})
    if category.get("available"):
        lines.append(f"  category: {category['label']} ({category['probability']:.2f}) "
                     f"| confidence: {alert['confidence']['alert_probability']:.2f}"
                     f"{'' if alert['confidence']['calibrated'] else ' (uncalibrated)'}")
    for phrase in alert.get("audio_evidence", {}).get("descriptions", [])[:2]:
        lines.append(f"  audio: {phrase}")
    acoustic = alert.get("audio_evidence", {}).get("acoustic_events", {})
    if acoustic.get("available") and acoustic.get("segments"):
        top = max(acoustic["segments"], key=lambda s: s["incident_energy"])
        if top["events"]:
            names = ", ".join(f"{e['label']} ({e['probability']:.2f})" for e in top["events"][:3])
            lines.append(f"  sound (YAMNet, corroboration): {names}")
        else:
            lines.append("  sound (YAMNet): no incident-relevant event named")
    for frame in alert.get("media", {}).get("key_frames", [])[:3]:
        lines.append(f"  frame @{frame['second']}s -> {frame['path']}")
    for key in ("spectrogram",):
        if alert.get("media", {}).get(key):
            lines.append(f"  {key} -> {alert['media'][key]}")
    return "\n".join(lines)


def category_names() -> tuple[str, ...]:
    """Exposed so the UI and tests iterate the same class order the model was trained with."""
    return CATEGORIES

