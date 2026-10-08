"""Losses for weakly-supervised MIL training, plus masking helpers.

All losses are defined on *snippet* score logits and clip-level targets, because XD-Violence only
provides weak (clip-level) labels. Three terms are combined:

1. ``topk_mil_loss``   - the core MIL objective: the mean of the k highest snippet probabilities is
   compared with the clip label. Rationale: a clip is positive if *at least one* moment is
   anomalous, and the mean of the top-k is a differentiable, less brittle surrogate than the max.
2. ``attention_loss``  - BCE on the attention-pooled clip score (gives the attention weights used as
   temporal evidence in the explanation panel).
3. ``multilabel_loss`` - BCE per category on the 6 XD-Violence classes (multi-label: a clip can be
   riot *and* explosion, so sigmoid + BCE, never softmax).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-7


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Softmax over `dim` ignoring positions where ``mask`` is False (no NaNs from -inf rows)."""
    neg_inf = torch.finfo(scores.dtype).min
    scores = torch.where(mask, scores, torch.full_like(scores, neg_inf))
    weights = torch.softmax(scores, dim=dim)
    return torch.where(mask, weights, torch.zeros_like(weights))


def topk_mil_loss(snippet_logits: torch.Tensor, mask: torch.Tensor, binary: torch.Tensor,
                  k: int = 5) -> torch.Tensor:
    """Mean of the k highest snippet probabilities vs the clip-level label."""
    probs = torch.sigmoid(snippet_logits)
    neg_inf = torch.finfo(probs.dtype).min
    masked = torch.where(mask, probs, torch.full_like(probs, neg_inf))
    k_eff = max(1, min(k, int(mask.sum(dim=1).min().item())))
    clip_prob = masked.topk(k_eff, dim=1).values.mean(dim=1).clamp(EPS, 1 - EPS)
    return F.binary_cross_entropy(clip_prob, binary)


def attention_loss(clip_logits: torch.Tensor, binary: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(torch.sigmoid(clip_logits).clamp(EPS, 1 - EPS), binary)


def multilabel_loss(multi_logits: torch.Tensor, multi: torch.Tensor,
                    pos_weight: torch.Tensor | None = None,
                    abuse_hard_negative_weight: float = 1.0) -> torch.Tensor:
    """Per-category BCE on probabilities.

    ``pos_weight`` reweights the positive term per category (``nn.BCELoss(pos_weight=...)``
    semantics). It exists because XD-Violence's six classes are wildly unbalanced: ``abuse`` has 50
    positive clips out of 3 804 in train against 367-463 for every other class, and the unweighted
    head simply never fires (measured: 0 TP / 0 FP at threshold 0.5, ``docs/journal.md`` Session 10
    addendum).

    The weight is applied by hand rather than with ``F.binary_cross_entropy_with_logits``: the
    probability form is what every checkpoint so far was trained with, and switching forms would
    move the baseline as well as the experiment. ``w = pos_weight`` on positives and 1 elsewhere is
    exactly what the logits form does with ``pos_weight`` (verified against it in the tests).
    """
    probs = torch.sigmoid(multi_logits).clamp(EPS, 1 - EPS)
    element_loss = F.binary_cross_entropy(probs, multi, reduction="none")
    weight = torch.ones_like(multi)
    if pos_weight is not None:
        weight = weight + multi * (pos_weight - 1.0)
    if abuse_hard_negative_weight > 1.0:
        # CATEGORIES order is fighting, shooting, riot, abuse, car_accident, explosion.
        # Clips labeled fighting/shooting but not abuse are valuable hard negatives for the
        # abuse head. Keep this weighting limited to the abuse output so the other category
        # heads and positive abuse co-occurrences retain their original loss.
        abuse_negative = multi[:, 3] < 0.5
        violence_label = (multi[:, 0] > 0.5) | (multi[:, 1] > 0.5)
        hard_negative = abuse_negative & violence_label
        weight[:, 3] = torch.where(
            hard_negative,
            weight[:, 3] * abuse_hard_negative_weight,
            weight[:, 3],
        )
    return (element_loss * weight).mean()


def intra_video_contrastive_loss(snippet_logits: torch.Tensor, mask: torch.Tensor,
                                 binary: torch.Tensor, k: int = 5,
                                 margin: float = 0.5) -> torch.Tensor:
    """Intra-clip contrastive separation penalty for positive (abnormal) clips.

    Forces peaceful/background moments within an anomalous video to stay close to 0,
    sharpening temporal onset and offset boundaries.
    """
    probs = torch.sigmoid(snippet_logits)
    pos_idx = (binary > 0.5).nonzero(as_tuple=True)[0]
    if len(pos_idx) == 0:
        return torch.tensor(0.0, device=snippet_logits.device)

    pos_probs = probs[pos_idx]
    pos_mask = mask[pos_idx]

    neg_inf = torch.finfo(probs.dtype).min
    pos_inf = torch.finfo(probs.dtype).max

    masked_top = torch.where(pos_mask, pos_probs, torch.full_like(pos_probs, neg_inf))
    masked_bot = torch.where(pos_mask, pos_probs, torch.full_like(pos_probs, pos_inf))

    k_eff = max(1, min(k, int(pos_mask.sum(dim=1).min().item())))
    top_means = masked_top.topk(k_eff, dim=1).values.mean(dim=1)
    bot_means = (-masked_bot).topk(k_eff, dim=1).values.neg().mean(dim=1)

    loss = F.relu(margin - (top_means - bot_means)).mean()
    return loss


def total_loss(outputs: dict, batch: dict, k: int = 5, w_attn: float = 0.5,
               w_multi: float = 0.3, w_contrastive: float = 0.0,
               multi_pos_weight: torch.Tensor | None = None,
               abuse_hard_negative_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Weighted sum of the loss terms; also returns the individual values for logging."""
    mask = batch["mask"]
    mil = topk_mil_loss(outputs["snippet_logits"], mask, batch["binary"], k=k)
    attn = attention_loss(outputs["clip_logits"], batch["binary"])
    multi = multilabel_loss(outputs["multi_logits"], batch["multi"], multi_pos_weight,
                            abuse_hard_negative_weight=abuse_hard_negative_weight)
    
    total = mil + w_attn * attn + w_multi * multi
    parts = {
        "loss": float(total.detach()),
        "loss_mil": float(mil.detach()),
        "loss_attn": float(attn.detach()),
        "loss_multi": float(multi.detach()),
    }
    if w_contrastive > 0.0:
        contra = intra_video_contrastive_loss(outputs["snippet_logits"], mask, batch["binary"], k=k)
        total = total + w_contrastive * contra
        parts["loss"] = float(total.detach())
        parts["loss_contrastive"] = float(contra.detach())

    return total, parts
