# SafeWatch protocol — what we follow, what we changed, and how numbers may be compared

> Draft material for the *méthodologie* chapter. Every rule here is either inherited from the benchmark
> or a deliberate, documented deviation. Companion docs: `docs/dataset.md` (measured data facts),
> `docs/theory_notes.md` (T1–T10), `docs/journal.md` (what actually ran).

Three layers must be kept apart, because changing one silently changes reported numbers:

| Layer | Definition | Our position |
|---|---|---|
| **Data protocol** | which clips train, which are evaluated, what labels exist | **inherited from XD-Violence** (Wu et al., ECCV 2020) — not our choice to change |
| **Metric protocol** | how predictions are scored and aggregated | **upgraded** relative to the 2020 paper (see §2) |
| **Method protocol** | architecture, supervision trick, fusion, robustness training | **modern** (2024–2025 references, see §3) |

---

## 1. Data protocol (inherited)

- **Dataset**: XD-Violence only. No other dataset is used anywhere in the project.
- **Splits**: 3 950 train clips / 800 test clips (mirror counts; the paper reports 3 954 — a 4-clip
  discrepancy of the mirror, documented in `docs/dataset.md`).
- **Supervision**: training uses **clip-level weak labels only** (binary + 1–3 categories encoded in the
  file name). Frame-level annotations exist **only for the test split** and are never used for model
  selection.
- **Validation split**: 400 clips carved out of train, stratified by first category, fixed seed 42
  (`scripts/build_lists.py`). Model selection, thresholds and segment parameters are chosen there.
- **Features**: precomputed, frozen. Mirror track: `swin_rgb` 768-d primary, `i3d_rgb` 2048-d as an
  ablation. Official track: the authors' 1024-d I3D RGB features (16-frame grid), obtained from the
  released archive and used for one visual-only run — **the mirror and official tracks are scored
  against their own clip lists and grids and are never mixed in one table**
  (`runs/.../test_metrics.json` records `stride_frames`, `config.json` records `feature_set`).
  Audio: log-mel patch statistics (132-d) extracted locally; VGGish 128-d exists for a 10-clip subset
  only (CPU budget), so no VGGish-based number is quoted.
- **Temporal grid**: **64 frames per snippet = 2.667 s at 24 fps**, i.e. `T = ceil(nb_frames / 64)`.
  Verified on 10 clips from both sources (movie and YouTube), all 24.000 fps
  (`scripts/check_feature_grid.py`). Frame annotations are converted to snippet indices with this rule;
  never assume the papers' 16-frame grid.
- **Test ground truth**: 500 clips carry 1 238 intervals; the other 300 are normal ⇒ all 800 clips are
  scoreable.

## 2. Metric protocol (upgraded)

The 2020 paper reports a single **global frame-level PR-AUC** and no localisation metric. We report:

1. **Both aggregations** — `global` (one PR curve over all pooled snippets; Wu-compatible) and
   `per_video` (mean of the 800 per-clip APs; the modern headline used by VadCLIP/MSBT/MAVD/HyperVD).
2. **Both metric functions** — `pr_auc` (`auc(recall, precision)`, the official code's recipe) and
   `average_precision` (tie-safe step-interpolated AP). Measured divergence with tied scores:
   prevalence 0.5 gives 0.75 vs 0.50 — both are therefore always printed side by side.
3. **Localisation quality** — segment-level precision/recall/F1 at tIoU 0.1…0.5, mean best-IoU, and
   detection delay (the subject requires start *and* end of the incident; the benchmark papers omit it).
4. **Per-class AP** for the 6 categories (clip level, from the multi-label head), with the sample size
   stated per class (`abuse` has only 8 test clips → wide error bars).
5. **Calibration** — reliability of the reported confidence (temperature/isotonic fit on validation,
   ECE reported), because every alert shows a confidence percentage.
6. **No point-adjustment.** Scores are evaluated as produced; point-adjusted variants (which give credit
   for a whole event once one moment is found) inflate numbers and are not used.

## 3. Method protocol (modern)

- **Baseline (P1)**: visual-only MIL — Conv1d temporal encoder, masked temporal attention, top-k MIL loss
  (`k`=5) + attention BCE + multi-label BCE. Reproduces the *spirit* of Wu et al. without their graph
  branches, and establishes the re-baselined reference for this feature set.
- **Fusion ladder (P2/P3)**, each rung measured as an ablation row: early fusion (feature concat) →
  late fusion (score averaging) → cross-attention (mutual refinement) → **adaptive gating** with
  reliability signals (video brightness/blur/occlusion, audio SNR/flatness/VAD, model disagreement).
- **Alignment before fusion** (MAVD, KBS 2025): map audio/flow features into the video semantic space
  before mixing, instead of relying on naive concatenation.
- **Robustness training** (AVadCLIP, 2025): modality dropout during training (p≈0.15) so the model
  copes with a missing modality; optional uncertainty-driven distillation toward a unimodal student.
- **Robustness evaluation**: AP as a function of audio SNR, video brightness/blur/occlusion, and with a
  modality removed — the subject's "rester fonctionnelle lorsqu'une modalité est dégradée".
- **Localisation**: class-specific temporal attention (T-CAM lineage: UR-DMU / PEL4VAD) so the UI can
  show *when* each category fires, not just one global score curve.
- **Segment rule in seconds, not snippets** (Session 9): `min_length` in snippets silently means a
  different duration per grid (2 snippets = 5.3 s at 64 frames, 1.3 s at 16). Any published segment
  number goes through `metrics.snippets_for_seconds()` and both units are recorded in
  `test_metrics.json` under `segment_rule`.
- **Reliability signals (adaptive fusion)**: seven channels - audio `rms`, `rms_db`, spectral
  `flatness`, `zcr`, `clip_ratio` pooled on the snippet grid, plus visual `feature_energy` and
  `temporal_motion` computed from the frozen features. Standardised with **train-split** statistics
  only (`scripts/quality_stats.py`); a modality that is degraded or absent is encoded by the sentinel
  `-5.0` on its channels. The visual channels are *proxies*: with frozen features we cannot measure
  true brightness/blur, and we say so instead of claiming we do.
- **Robustness protocol**: one function (`reliability.drop_modality`) performs the degradation, and it
  is the same one used by training dropout (`--modality-dropout`, default 0.15) and by scoring
  (`--drop-modality audio|visual`). A modality-removal number quoted in the report therefore measures
  exactly what the model was trained to expect.
- **Online mode**: a causal variant of the head (HL-Net's approximator idea) to quantify the offline →
  online AP cost instead of claiming streaming is free.

## 4. Comparability rules (for every table in the report)

| Number | May be compared with published values? |
|---|---|
| Ours, `per_video` AP on `swin_rgb` features | **No** — different features and grid; state it explicitly |
| Ours, `global` `pr_auc` | Only as a *protocol* reference to Wu et al.'s definition, not as a value |
| Our visual-only vs our audio-only vs our fusion | **Yes** — same features, same splits, same code ⇒ the ablation is valid |
| Published SOTA rows | Quote as context with their feature set and protocol annotated |

Rule: never place our number next to a published number without annotating the feature set, the
temporal grid, the aggregation and the metric function.
