# Theory notes — what you need to know, and how deep

> Companions: `docs/reading_plan.md` (what to read, in which order) and `docs/journal.md`
> (what we measured on the real data).
> **Rule: learn a topic just before the code needs it.** Theory read with no immediate use is
> forgotten; theory applied in the same week sticks.

## Three levels — stop at the right one

| Level | You can… | Needed for |
|---|---|---|
| **A. Implement** | derive it on paper, implement it, debug it | MIL loss, metrics, fusion module, ground-truth alignment |
| **B. Use** | understand inputs/outputs, tune the knobs, justify the choice | NN training, multi-label heads, quantisation, streaming |
| **C. Read-only** | understand abstract + figure well enough to cite and position our work | CLIP-style contrastive learning, GCNs, hyperbolic embeddings, prompt methods |

Most study plans fail by treating level C as level A. We do the opposite: A properly, B practically,
C quickly.

## Work order (phase → topics → time box)

| Phase | Topics | Level | Time |
|---|---|---|---|
| P1 — visual baseline | T1 MIL, T2 metrics, T3 NN fundamentals | A / A / B | 8 h |
| P2 — audio + fusion | T4 attention, T5 fusion & robustness, T6 features & audio DSP | A / A / B | 10 h |
| P3/P4 — localisation | T1 revisited (top-k), T7 multi-label & imbalance | A / B | 2 h |
| P5 — explanations | T8 attribution & its limits | B | 2 h |
| P7 — edge | T9 quantisation, T10 streaming/causal inference | B | 4 h |
| all phases | the three traps (Appendix) | A | 1 h, re-read often |

**Total ≈ 27 h of targeted study** across a 2–3 month project, i.e. ~2–3 h per week. That is the
honest budget — not a semester of optimisation theory.

---

## T1 — Weak supervision & Multiple Instance Learning (MIL) — **Level A**

**Idea.** You have a label for the whole clip ("violence / not", plus 1–3 categories) but none for
the individual moments. Model the clip as a **bag** and each snippet as an **instance**; the bag is
positive if at least one instance is positive. Since "at least one" is not usefully differentiable,
it is replaced by a smooth surrogate:

```
score(clip) = mean of the k largest snippet scores     # top-k pooling (k ~ 3-10)
loss        = BCE(score(clip), clip_label)             # binary cross-entropy
```

Everything in weak-supervision video anomaly detection is a variation on this: the value of `k`, the
pooling rule (max / top-k / attention-weighted), and how snippet features are contextualised before
pooling.

**Why this project forces it.** Training labels really are clip-level only — verified in
`docs/dataset.md`: the mirror contains no training annotation file, and labels live in the file names
(multi-label, `0`-padded).

**Where it lives here.**
- `safewatch/eval/metrics.py::snippet_labels()` — the evaluation-side twin (frame intervals → per-snippet labels).
- Training: the MIL loss to be written in P1 (`safewatch/losses/mil_topk.py`).
- Reference: HL-Net (ECCV 2020) Method section + loss; `XDVioDet/{model,train}.py` (we already read
  them: `classifier`, `approximator`, `BCELoss`, `is_topk=True`).

**Self-check.**
1. Why does top-k beat mean pooling when only ~3 % of snippets are anomalous?
2. A clip is labelled *fighting* but contains 40 s of normal footage — what stops the model from
   scoring that normal stretch high?
3. At test time, is the clip score or the snippet score compared with the ground truth? Why does the
   official code need `np.repeat(pred, 16)`?

**Time box:** 3–4 h (paper + code + deriving the loss by hand).

---

## T2 — Detection metrics: PR/AP, the two protocols, calibration — **Level A**

**Idea.** Accuracy is useless here: ~60 % of test *clips* are violent but anomalies are sparse inside
each one, and a trivial "always anomalous" predictor would score 0.6 accuracy while being worthless.
So we score a **ranking** with the precision-recall curve, summarised as **AP** (area under it):

- `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)` computed at every threshold of the score;
- **AP** averages precision over recall levels → rewards ranking anomalies above normal snippets.

Two aggregation protocols exist and are **not interchangeable**:
- `global` — one PR curve over all snippets of all 800 test clips pooled (what the official XDVioDet
  code does; HL-Net = 0.7864);
- `per_video` — AP computed per clip, then averaged (what VadCLIP/MSBT/MAVD report).
Always report both, and say which one a comparison uses.

**The tie artefact (measured in our repo, not folklore).** With constant scores and prevalence 0.5,
`average_precision` returns **0.50** (tie-safe) while the official `pr_auc` returns **0.75** — a
trapezoidal-integration artefact. Consequence: any number produced by a *constant* or heavily tied
scorer is inflated under the official protocol. Our harness therefore reports both
(`safewatch/eval/metrics.py::pr_auc`, `::average_precision`, verified by `tests/test_metrics.py`).

**Calibration** is a separate question: the score is a **probability** only after calibration
(temperature scaling / isotonic regression on the validation split). "Confiance 87 %" in the alert is
meaningless unless the curve is calibrated — plot reliability diagrams, report ECE.

**Localisation metrics** (the subject's "position précise"): IoU between predicted and annotated
segments, P/R/F1 at tIoU thresholds (0.1…0.5), greedy highest-IoU-first matching, and detection delay
(snippets between annotation start and prediction start). Implemented:
`iou`, `_greedy_match`, `localization_report`.

**Where it lives here.** `safewatch/eval/metrics.py`, `tests/test_metrics.py`, and the "AP" definition
inside the official `XDVioDet/test.py`. Thresholds used for segmentation must be chosen on the
**validation** split, never on test.

**Self-check.**
1. A model scores every snippet with the clip-level label it predicted. Which protocol punishes it
   more, and why?
2. Why is AP ≥ AUROC misleading when prevalence is 0.03?
3. Our stride is 63 frames: what is the smallest localisation error you can even express, and how
   does that bound `f1@iou0.5` for short events?

**Time box:** 3 h (including hand-computing a PR curve on 10 snippets).

---

## T3 — Deep-learning training fundamentals — **Level B**

**Idea.** Everything you need to debug a run, nothing more:

- **Loss**: `BCE = -[y·log p + (1-y)·log(1-p)]`; it is what `torch.nn.BCELoss` computes on sigmoid
  outputs. Multi-label heads use BCE per class (sigmoid), *not* softmax (which forces exclusivity —
  wrong here, since a clip can be riot **and** explosion).
- **Optimiser**: Adam with lr ~1e-4, weight decay 0 (the reference baseline's choice), a
  `MultiStepLR` dropping ×0.1 around epoch 10, and early stopping on the validation AP.
- **Regularisation**: dropout 0.6 (the reference uses this on every block), top-k MIL sparsity,
  and a 400-clip validation split. Overfitting signature: val AP rises then falls while train loss
  keeps dropping.
- **Practical facts for our setup**: features are *precomputed and frozen*, so the model is tiny
  (ours: 656 k params) and one training step costs ~24 ms measured on this CPU ⇒ an epoch is seconds.
  That means you can afford many runs — use them for ablations rather than for exotic architectures.

**Where it lives here.** The training engine to be written in P1 (`safewatch/train/engine.py`),
plus our measured benchmark in `scripts/smoke_test.py`.

**Self-check.**
1. Why does the reference code use a *lower* learning rate for the approximator branch
   (`args.lr/2`)? What does that tell you about the two objectives?
2. Your val AP stalls at prevalence (≈0.5 for the pooled protocol). Name three likely causes and the
   cheapest diagnostic for each.
3. Why is batch size 128 with 200-snippet sequences cheap here but impossible for end-to-end video?

**Time box:** 4 h (mostly by doing: train the P1 baseline and watch the curves).

---

## T4 — Sequence modelling & attention (incl. cross-modal attention) — **Level A**

**Idea.** Three ways to give a snippet access to its context:

- **Temporal convolution** (1D conv over time): local, cheap, fixed receptive field (kernel 5 ⇒ ±2 snippets).
- **Self-attention**: every snippet attends to every other → long-range (“the whole clip's rhythm”),
  cost O(T²).
- **Cross-attention** (the fusion tool): queries from one modality, keys/values from the other, so
  audio can *select* which visual moments matter and vice versa.

Shapes for our case (memorise these, they prevent 80 % of debugging time):

```
video V ∈ R^{B×T×dv},  audio A ∈ R^{B×T×da}          # same T, aligned by the snippet grid
Q = W_q V, K = W_k A, V' = W_v A                     # B×T×d each
attn = softmax(Q Kᵀ / √d) ∈ R^{B×T×T}
out  = attn · V'                                     # B×T×d  → video refined by audio
```

Multi-head = repeat with `d/h` per head and concatenate. Without positional information, a
permutation-invariant layer cannot tell "before" from "after" — that matters because causality
(streaming) and order (collision *then* braking) carry meaning.

**Why this project forces it.** "Attention croisée" is explicitly in the subject, and the temporal
attention weights are also the cheapest honest source of *temporal* evidence for the explanation panel.

**Where it lives here.**
- `safewatch/models/fusion.py` (to be written in P2/P3);
- reference implementations: `cma_xdVioDet/model.py` (audio-guided attention, small file) and
  HL-Net's three branches — similarity prior (≈ self-attention), locality prior (≈ positional decay),
  score prior (≈ attention over predicted scores). Same idea, three different priors.

**Self-check.**
1. Write the shapes for cross-attention where audio attends to video and T=200, d=128, h=4.
2. Why is O(T²) acceptable here (T≤~300) but not for 1080p pixels?
3. If you shuffle the snippet order before fusion, which parts of the pipeline break and which don't?

**Time box:** 4 h (2 h reading, 2 h implementing attention over random tensors until shapes are obvious).

---

## T5 — Multimodal fusion & modality robustness — **Level A**

**Idea.** The subject names four strategies; you must implement and *compare* them:

| Strategy | How | Failure mode |
|---|---|---|
| Early fusion | concatenate features `[V;A]` → one encoder | a noisy/dead modality poisons everything |
| Late fusion | separate encoders, average the scores | cannot exploit cross-modal cues (impact + collision) |
| Cross-attention | mutual refinement (T4) | needs aligned T; heavier |
| **Adaptive (ours)** | `z(t) = α(t)·V(t) + (1-α(t))·A(t)` with `α(t)` from *reliability* | α needs calibration, else it hides a broken modality |

Reliability signals you can actually compute cheaply (this is what makes "adaptive" honest rather
than cosmetic): video — mean brightness, Laplacian-variance blur, occlusion ratio, motion energy
percentile; audio — estimated SNR, spectral flatness, VAD activity, clipping ratio; model — agreement
between the unimodal branches, MC-dropout variance, attention entropy.

**Robustness training** = **modality dropout**: during training, zero out one modality (p≈0.15) so the
model never assumes both are present. Then the degraded-modality curves are an *evaluation* of
something you trained for, not a surprise. The stronger variant is AVadCLIP's **uncertainty-driven
distillation** (§III-F): a visual-only student imitates the audio-visual teacher, weighted by how hard
(uncertain/diverse) each sample is — that is what makes a unimodal deployment competitive.

**Why this project forces it.** "Rester fonctionnelle lorsqu'une modalité est dégradée ou
indisponible" is a graded requirement, and it is the sentence that distinguishes this project from a
plain late-fusion baseline.

**Where it lives here.** `safewatch/models/fusion.py`, `safewatch/data/degrade.py` (simulating dark
video / noisy audio), `safewatch/eval/robustness.py` (AP vs SNR, brightness, modality-dropped).
Reference: AVadCLIP §III-C + §III-F + ablations §IV-D1/§IV-D3.

**Self-check.**
1. Your model ignores α and always weights video 0.8. How does the dark-video curve expose that?
2. Modality dropout versus training on complete data only: why does the former *not* hurt clean-input AP?
3. How would you prove that the gate uses reliability and not just a constant learned bias?

**Time box:** 4 h (2 h concepts + 2 h designing the reliability features and the degradation protocol).

---

## T6 — Frozen features & audio DSP — **Level B**

**Idea.** Instead of training a video/audio network (impossible on this CPU), we consume **frozen
embeddings** from models pretrained on big datasets, and train only a small head on top. This is
transfer learning in its cheapest form; it is standard in this literature (every XD-Violence paper
starts from I3D/VGGish features).

- **I3D (Kinetics-400) RGB** → 2048-d per snippet. The mirror gives 5 crops per snippet
  (0 = centre, 1–4 = corners); averaging them is the safe default, crop-0-only the cheap variant.
- **VGGish** → 128-d per 0.96 s patch. Input contract: **16 kHz mono**, 64-mel, 96-frame log-mel patch.
  Our audio is **48 kHz and 6-channel** (measured) ⇒ downmix + resample before anything else.
- Alignment: a snippet is ~2.63 s ⇒ **~2.75 VGGish patches per snippet** ⇒ average-pool patches onto
  the snippet grid. Getting this wrong shifts audio relative to video by seconds and quietly destroys
  fusion (the classic silent failure).
- Domain gap: Kinetics/AudioSet ≠ surveillance. It costs accuracy, but it buys the ability to train
  in minutes instead of weeks on this machine. Say so explicitly in the report.

**Where it lives here.** `docs/dataset.md` (measured facts), `scripts/extract_audio_features.py`
(to be written), `safewatch/features/`.

**Self-check.**
1. Why is 16 kHz used for VGGish although the source is 48 kHz? What is lost?
2. Our stride is 63 frames: how many VGGish patches per snippet, and what pooling do you choose?
3. Name one failure the 5-crop average hides that crop-0 keeps.

**Time box:** 3 h.

---

## T7 — Multi-label classification & class imbalance — **Level B**

**Idea.** XD-Violence assigns 1–3 categories per violent clip (they co-occur: an explosion inside a
car accident). Therefore:

- **sigmoid + BCE per class**, never softmax (softmax would force mutual exclusivity);
- **per-class AP**, plus a macro average that treats the 6 categories equally — note that
  `abuse` has only **8 test clips**, so its per-class AP is statistically fragile and must be
  reported with that caveat, not hidden;
- **per-class thresholds** for the UI's category scores; a single shared threshold will always favour
  the frequent classes (fighting 120, riot 101, car_accident 96, explosion 91, shooting 84, abuse 8);
- **class-specific temporal attention** is how you get "where in the clip was *this* category" from
  weak labels (T-CAM), which the UI needs when several categories fire.

**Where it lives here.** The category head + T-CAM in `safewatch/models/` (P3/P4);
`docs/dataset.md` holds the exact per-class test counts.

**Self-check.**
1. A clip is riot+explosion. What does BCE do that softmax+CE cannot?
2. Should we weight the loss by inverse class frequency? Argue both sides, then decide from the
   validation AP per class.
3. With 8 test clips for `abuse`, how do you report its AP honestly?

**Time box:** 2 h.

---

## T8 — Attribution & its limits — **Level B**

**Idea.** Three attribution tools, each with different trustworthiness:

| Tool | What it gives | Caveat |
|---|---|---|
| Temporal attention weights | which snippets influenced the clip score | attention is *evidence*, not proof (Jain & Wallace 2019) |
| Grad-CAM on the light frame encoder | *where* in a frame the model looked | needs gradients; coarse at stride > 1 |
| Leave-one-modality-out (LOO) | how much each modality contributed to a *specific* score | expensive (2 forward passes), but the only causal measure here |

LOO formula for the alert: `contribution_video = s(both) − s(audio only)`,
`contribution_audio = s(both) − s(video only)`; report both plus the adaptive gate α(t) as the
cheap, per-snippet version. Never present attention as causality — that sentence belongs in the
report's limitations section, and in the operator's mental model.

**Where it lives here.** `safewatch/explain/` (P5) and the alert JSON (`modality_contribution` field).

**Self-check.**
1. Give a case where Grad-CAM highlights the lamp because the lamp correlates with violence.
2. Why can attention be high on a snippet that a LOO test shows is irrelevant?
3. Which of the three tools would you trust to justify a *rejection* by an operator?

**Time box:** 2 h.

---

## T9 — Quantisation for the Edge — **Level B**

**Idea.** "Optimised model" for a constrained device means smaller and faster, with the least accuracy
lost. For a PyTorch CPU pipeline the practical ladder is:

| Step | What changes | Cost |
|---|---|---|
| ONNX export (fp32) | framework overhead removed, graph fused | no accuracy change |
| **Dynamic** int8 | weights stored int8, activations quantised at run time, **no calibration data** | usually < 1 % metric change |
| Static int8 (QDQ) | activations calibrated on a small representative set | tighter latency, needs calibration |
| fp16 / TensorRT (Jetson) | GPU-specific engines | needs the hardware |

Two facts that decide where to spend effort: (i) **convolutions and linear layers quantise well,
softmax/attention poorly** — so quantising the tiny head saves little; the win is in the *feature
extractor* of the edge track; (ii) measure **latency and peak RSS**, not only file size, because
"quasi réel" is a latency claim (`temps de traitement < durée du flux`).

**Where it lives here.** `safewatch/edge/{export_onnx,quantize,bench_latency}.py` (P7) and
`docs/edge_deployment.md`. The honest column "verified here / not verified without Jetson" is part of
the deliverable.

**Self-check.**
1. Why does dynamic quantisation need no calibration set, and what does it therefore not quantise?
2. Our head has 656 k params; the light extractor has millions. Which one do you quantise first, and how
   would you show the benefit?
3. What does "quasi réel" mean precisely for a 25 fps stream: which number must stay above what?

**Time box:** 2 h theory + 2 h measuring (P7).

---

## T10 — Streaming / causal inference — **Level B**

**Idea.** Offline mode sees the whole clip; online mode sees only the past. Concretely:

- features arrive per snippet (~2.63 s of content per feature vector in the benchmark track);
- the model scores a **sliding window** (e.g. the last 32 snippets) so it can use recent context
  without rewriting the whole clip — this is what HL-Net's `approximator` branch approximates;
- **latency = window fill time + inference time**; an alert for a still-open segment is *pending* and
  gets finalised when the score drops below the threshold;
- causality costs accuracy: the reference reports 0.7864 offline vs **0.7553** online — quote that
  number rather than pretending streaming is free;
- a ring buffer + a monotonic clock is all the infrastructure needed; no exotic streaming framework.

**Where it lives here.** `safewatch/serving/{ingest,pipeline}.py` (P6) and
`safewatch/models/` (the causal variant of the head, P4).

**Self-check.**
1. Why can the offline model be better than any causal model, by construction?
2. For our measured 24 ms per training step, where is the real latency bottleneck in a live pipeline?
3. How do you avoid emitting 12 alerts for one continuous event?

**Time box:** 2 h.

---

## Appendix — the three traps that silently invalidate a project

### Trap 1 — metric conventions and ties
What happens if you miss it: you report a number nobody can reproduce, and comparisons are wrong.
Evidence: with constant scores, prevalence 0.5 → `average_precision` = **0.50**, official `pr_auc` =
**0.75** (`tests/test_metrics.py`).
Prevention: always report **both protocols × both metrics**, and state which one a claim uses.

### Trap 2 — stride / ground-truth alignment
What happens if you miss it: AP collapses to chance with no error message.
Evidence: the mirror's features sit on a ~**63.3-frame** grid (`T = ceil(nb_frames/64)`), not the
official 16-frame grid; annotations are in frames (`docs/dataset.md`).
Prevention: `snippet_labels()` takes the measured float stride; never hard-code 16.

### Trap 3 — leakage
What happens if you miss it: model selection on test ground truth inflates every reported number.
Evidence: 500 of 800 test clips carry frame-level intervals.
Prevention: a 400-clip validation split carved from train, scored with **clip-level** labels only;
test GT touched once at the end.

---

## What you do *not* need

Hyperbolic geometry (HyperVD), optimal transport, prompt-engineering internals (VadCLIP), GCN
derivations (we use cross-attention instead), diffusion models, compiler/TensorRT kernel internals,
codec internals beyond ffmpeg flags, and any novel-theory contribution. If a topic is not in T1–T10,
it is *citable* but not *required*.

---

## Six-week study schedule (≈2–3 h/week, aligned with the phases)

| Week | Study | Produces |
|---|---|---|
| 1 | T1, T2 + P1 reading block | the baseline note in the journal + first AP |
| 2 | T3, T6 + P2 reading block | audio features + reliability formula |
| 3 | T4, T5 + P3 reading block | fusion module + ablation matrix |
| 4 | T7 + P4 reading block | segment extraction rules + localisation table |
| 5 | T8 + P5 reading block | alert JSON + evidence figures |
| 6 | T9, T10 + P6/P7 blocks | UI + edge table + report skeleton |

**You are ready when you can, without notes:** (1) write the MIL loss, (2) explain both AP protocols
and the tie artefact, (3) draw cross-attention shapes on a whiteboard, (4) justify α(t) from
reliability features, (5) convert a score curve into segments with validated parameters, and (6) say
which of our numbers are comparable with published ones and which are re-baselined.




