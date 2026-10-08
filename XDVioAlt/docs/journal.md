# SafeWatch Edge AI — journal de bord / decision log

One entry per working session: what was run, what came out, what was decided, what is blocked.
This replaces Git history (no repository by choice).

---

## 2026-09-23 — Session 1 — Environment + skeleton

**Goal:** working Python env with CPU PyTorch, plus project skeleton.

**Commands run**
```bash
cd /home/aarf101/projtutore
mkdir -p data/{raw,features,lists,annotations} safewatch/{data,eval} scripts notebooks tests docs runs
python3 -m venv .venv
.venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install numpy pandas scikit-learn matplotlib seaborn tqdm pyyaml \
    opencv-python-headless librosa soundfile onnx onnxruntime plotly streamlit pytest ruff \
    jupyter ipykernel huggingface_hub
.venv/bin/python scripts/smoke_test.py
```

**Findings**
- `python3` → `/usr/bin/python3.12`, **but** the sandbox pre-created `.venv` from
  `/home/aarf101/.pyenv/versions/3.14.0` (pyenv). Result: `.venv` reports **Python 3.14.0** while
  `pyvenv.cfg` says `version = 3.12.3`. Consequence: **always use `.venv/bin/python`**; never rely on
  bare `python3`. Every dependency resolved for 3.14, so we accept the quirk.
- Verified stack (by import, not by hope): torch `2.14.0+cpu`, torchaudio `2.11.0+cpu`,
  onnxruntime `1.30.0` (`CPUExecutionProvider`), numpy `2.5.3`, pandas `3.0.6`, scikit-learn `1.9.1`,
  librosa `1.0.0`, soundfile `0.14.0`, opencv-python-headless `5.0.0.93`, onnx `1.23.0`,
  streamlit `1.64.0`, plotly `7.1.0`, matplotlib `3.11.2`, seaborn `0.13.2`, huggingface_hub `1.32.0`,
  pytest `9.1.1`, ruff `0.16.8`, PyYAML `6.0.3`, tqdm `4.70.1`.
- `torch.cuda.is_available()` = **False** (AMD iGPU only). `torch.get_num_threads()` = **6** while
  `nproc` = 12 → thread count is capped by the sandbox/cgroup. Revisit with `torch.set_num_threads(...)`
  and re-measure before any long training.
- ffmpeg / ffprobe `6.1.1` available.
- **CPU training feasibility measured**: conv1d fusion head (1152→512→128→1, 656 129 params),
  batch 8 × 200 snippets → **24.2 ms per training step** ⇒ ~3 s per 31-step epoch ⇒ tens of minutes for
  a 50-epoch run. CPU-only training of feature-based models is *not* a blocker.

**Decisions**
1. **No Git repository** (user choice). Reproducibility = `docs/journal.md` + per-run
   `runs/<date>_<exp>/` config snapshots. Adding Git later is a 2-command operation.
2. **Strategy A**: fully local, CPU-only; precomputed features from the HF mirror (I3D 2048-d) plus
   self-extracted VGGish audio from the mirror's raw videos. Cloud GPU bursts stay possible later.
3. Two tracks sharing one fusion head: benchmark track (precomputed features) and edge/demo track
   (light extractor + ONNX int8).
4. `.venv/bin/python` is the canonical interpreter for all commands and scripts.

**Artifacts created:** `README.md`, `requirements.txt`, `requirements-lock.txt` (157 lines),
`.gitignore`, `scripts/smoke_test.py`, `scripts/download_hf.py`, `docs/journal.md`, `docs/dataset.md`,
empty `safewatch/` package with `data/` and `eval/` sub-packages.

**Next session:** data bootstrap → `ffprobe` audio check on the sample videos → snippet-stride
calibration (the single most important number in the project).



---

## 2026-09-23 — Session 2 — Data bootstrap, audio check, stride calibration

**Goal:** obtain usable data + the ground-truth facts *before* writing model code.

**Commands**
```bash
.venv/bin/python scripts/download_hf.py --all --sample-videos 3     # meta + 800 test features + 3 clips
.venv/bin/python scripts/inspect_dataset.py --samples 7             # -> docs/dataset.md
```

**What was downloaded** (`data/raw/xd-violence/`, 1.5 GB)
- `data/i3d_rgb/test_videos/*.npy` — 800 files, `(T, 5 crops, 2048)` float32
- `data/test_annotations.txt` (500 clips), `data/test_list.txt` (800), `data/train_list.txt` (3950)
- 7 sample mp4 clips, one per category

**Findings (all measured)**
1. **Audio is present in 7/7 sampled clips** — AAC 48 kHz, **6–7 channels**. The mirror's videos are a
   usable audio source; VGGish must be fed mono 16 kHz audio (downmix + resample).
   ⇒ the biggest data risk (no audio, gated Baidu link only) is **resolved**.
2. **Clip id format**: `<film>__#HH-MM-SS_HH-MM-SS_label_<CODES>`; YouTube-sourced clips instead use
   `v=<youtube_id>__#<n>_label_<CODES>` (423 of the 800 test clips). `CODES` = `A` (normal) or
   `-`-joined categories with `0` padding (e.g. `label_B6-0-0`, `label_G-B2-B6`).
3. **Annotation coverage is complete**: 500 of 800 test clips carry 1 238 intervals (260 clips have
   ≥2 events); the 300 clips without intervals are **exactly the normal ones** (unannotated
   non-normal clips = 0). The full 800-clip test set is therefore scoreable with frame-level GT.
4. **Snippet stride ≈ 63 frames ≈ 2.63 s at 24 fps** (range 61.3–63.6) and the rule
   `T = ceil(nb_frames / 64)` holds for all 7 sampled clips across all 7 categories: the mirror's I3D
   features were re-extracted on a **64-frame non-overlapping window**, far coarser than the official
   16-frame grid. Feature dim 2048, 5 crops (crop 0 = centre).
5. Video: H.264, 24 fps, 640×346 typical → the demo/edge track is computationally light.
6. Scoring harness `safewatch/eval/metrics.py` + 13 tests pass; `ruff` clean. Measured metric quirk:
   with tied scores the two AP protocols diverge (`average_precision` = prevalence, official
   `pr_auc` = (prevalence+1)/2), so both are always reported.
7. Mirror quirk to avoid: its reference loader uses `str.rstrip('.mp4')`, which strips any of
   `{'.','m','p','4'}` and corrupts ids ending in `4` (e.g. `..._B4`); our parser uses `removesuffix`.

**Decisions**
- GT = per-snippet binary vectors built from `test_annotations.txt` using the **measured** stride
  (float, per clip); never assume 16.
- Benchmarks are **re-baselined on these features**: published AP values come from the official
  16-frame features, so absolute numbers are not directly comparable (we will always report our own
  visual-only baseline alongside).
- ~2.6 s localization granularity is too coarse for "01:24–01:31"-style alerts → the edge/demo track
  extracts its own features at ~1 s stride (needed for real time anyway).
- Audio pipeline: ffmpeg → mono/16 kHz PCM → log-mel → VGGish, average-pooled onto the snippet grid.

**Next session:** download full `i3d_rgb` (12 GB) for training, build split lists + a validation
subset, then the visual-only MIL baseline to obtain the first real AP number.

---

## 2026-09-23 — Session 3 — Theory guidance & reading plan (documents)

**Goal:** answer "does this project need theory?", and turn the answer into two study artifacts so the
theory is learned just-in-time instead of in the abstract.

**Created**
- `docs/theory_notes.md` (~410 lines) — ten topics T1–T10, each with *idea → why this project forces
  it → where it lives in our code/paper → self-check questions → time box*:
  T1 MIL/weak supervision, T2 metrics (both AP protocols, tie artefact, calibration, localisation
  metrics), T3 training fundamentals, T4 sequence modelling & cross-attention, T5 fusion & modality
  robustness, T6 frozen features & audio DSP, T7 multi-label & imbalance, T8 attribution & its limits,
  T9 quantisation, T10 streaming/causal inference. Plus an appendix on the three traps (metric ties,
  stride/GT alignment, leakage), a "what you do not need" list, and a six-week schedule (~27 h total,
  ≈2–3 h/week) mapped onto phases P1–P8.
- `docs/reading_plan.md` (~145 lines) — per-phase reading blocks with *why / stop-when criterion /
  experience level (A implement, B use, C cite) / time box*, an appendix reference table with links,
  and a "how to read a paper here" procedure.

**Design decisions (so the docs stay useful)**
1. Three experience levels — **A implement** (MIL, metrics, attention/fusion, GT alignment),
   **B use** (training knobs, quantisation, streaming, multi-label), **C cite only** (CLIP/contrastive
   learning, GCNs, hyperbolic, prompt methods). Most study plans fail by treating C as A; we invert that.
2. Every topic is anchored to something measurable in this repo (the 0.50 vs 0.75 tie artefact, the
   63.3-frame stride, the 8 `abuse` test clips, the measured 24 ms training step) rather than to
   generic textbook content.
3. Paper locations were verified rather than recalled: HL-Net arXiv:2007.04687 (three branches +
   approximator + ablations) and AVadCLIP arXiv:2504.04495 (§III-C adaptive fusion, §III-D prompts,
   §III-F uncertainty-driven distillation, §IV-D1/D3 ablations, §IV-E qualitative).
4. The Sci Rep 2025 reference is marked "cite with care": different setting (Ped2/Avenue, frame-level
   supervised AUC, no audio fusion) — it must not be presented as an XD-Violence baseline.

**Honest framing recorded in the docs:** the mathematics required is *undergraduate-level linear
algebra + probability*; the genuinely difficult part of this project is **experimental protocol
reasoning** (metric conventions, alignment, leakage), which is exactly what the appendix targets.

---

## 2026-09-23 — Session 4 — P1 pipeline built, feature grid resolved

**Goal:** code the P1 path (data → model → train → score) and settle the snippet-grid question that
every metric depends on.

**Created**
- `safewatch/data/dataset.py` — `ClipDataset`: reads the split CSVs, reduces the 5 crops (mean or
  centre), crops/subsamples/pads to a fixed length, returns features + padding mask + binary and
  multi-label targets. `max_snippets=None` yields full-length clips (inference).
- `safewatch/losses/mil.py` — `masked_softmax`, `topk_mil_loss` (core MIL objective), `attention_loss`,
  `multilabel_loss` (sigmoid+BCE per class), `total_loss` (weighted sum + per-term logging).
- `safewatch/models/mil.py` — visual-only baseline: Conv1d temporal encoder → masked temporal
  attention → per-snippet head (localisation + top-k MIL), clip head and 6-category head.
- `safewatch/train.py` — full training CLI whose argparse is generated from a `TrainConfig` dataclass;
  writes `runs/<date>_<tag>/{config.json, history.csv, metrics.json, ckpt_best.pt}`; model selection on
  the **validation split only** (clip-level labels).
- `safewatch/eval/runner.py` — scores the 800 test clips, converts frame annotations to snippet GT with
  the measured grid rule, reports **both AP protocols × both metrics**, the localisation report
  (F1@tIoU, mean best-IoU, detection delay) and clip-level per-class AP; saves `predictions.npz` +
  `test_metrics.json` next to the checkpoint.
- `scripts/build_lists.py` — writes `data/lists/{train,val,test}.csv` (stratified 400-clip validation
  split, seed 42) and `data/annotations/test_intervals.csv` (1 238 intervals, tidy form).
- `scripts/check_feature_grid.py` — probes clips and compares `nb_frames/T` vs `duration/T`.

**Verified (measured, not assumed)**
- Model: **542 601 params**, forward+backward+step = **17.8 ms** for batch 4 × 200 snippets ⇒ ~2.2 s per
  epoch at batch-128 equivalent. Masked attention rows still sum to 1 with padding; gradients flow.
- `ruff` clean on the whole repo; all four CLIs instantiate and print help.
- **Feature grid resolved**: 10 clips probed (7 movie-sourced + 3 YouTube-sourced) — **all 24.000 fps**,
  `nb_frames/T` ∈ [61.3, 64.0], and `T = ceil(nb_frames/64)` holds exactly. The mirror's features are on
  a **frame-based 64-frame grid = 2.667 s/snippet**, not a time-based one. `--stride-frames 64` is the
  default in the scorer; a per-clip override stays possible.

**Decisions**
- Feature set switched to **`swin_rgb` (VideoSwin 768-d, 4.5 GB)** as primary: verified drop-in
  compatible with `i3d_rgb` (same T, same 5-crop layout) but 2.7× smaller and a modern backbone;
  `i3d_rgb` (2048-d, 12 GB) remains available as an ablation row.
- **XD-Violence only**: the cross-dataset (CCTV-Fights) item is dropped from the plan.
- Downloads were launched with the `hf` CLI; `scripts/download_hf.py` still documents the mirror and its
  patterns (`--feature-set` support to be added when next touched).

**Next:** `build_lists.py` → `python -m safewatch.train --epochs 30` → `python -m safewatch.eval.runner`
⇒ the first re-baselined AP numbers for this feature set (expect ~0.70–0.80 for visual-only; audio and
fusion come in P2/P3).

---

## 2026-09-23 — Session 5 — First real numbers + two protocol defects found and fixed

**First complete P1 run** (`runs/2026-09-23_p1_visual_swin/`, visual-only, `swin_rgb`, 30 epochs,
test = 800 clips / 36 900 snippets):

| metric | value |
|---|---|
| global `pr_auc` (Wu-compatible) | **0.8100** |
| global `average_precision` | 0.8099 |
| per-video `pr_auc` | 0.7492 |
| per-video `average_precision` | **0.7709** |
| snippet-level prevalence (random floor) | 0.2586 |
| f1@tIoU 0.5 / 0.3 / 0.1 (threshold 0.5, untuned) | 0.322 / 0.456 / 0.605 |
| mean best IoU | 0.433 |
| mean detection delay | 4.69 snippets ≈ 12.5 s |
| per-class clip-level AP | riot 0.964 · car_accident 0.921 · fighting 0.805 · explosion 0.750 · shooting 0.549 · **abuse 0.065** (11 positives) |

**Defect 1 (code) — unit bug in the localisation report.** `mean_detection_delay_s` returned
`4.69 snippets × 64 frames = 300`, i.e. *frames* labelled as seconds. Fixed: the report now exposes
`mean_detection_delay_snippets`, `..._frames` and `..._seconds` (needs `fps`, now a CLI flag,
default 24). Real delay ≈ **12.5 s**.

**Defect 2 (protocol) — validation AP and test AP are not comparable.** Validation is measured on
**clip-level** binary labels (prevalence 0.523) and reached 0.9835; the test headline is **snippet-level**
localisation (prevalence 0.259) at 0.81/0.77. Different units, different floors ⇒ they must never be
placed side by side. Model selection stays on the clip-level validation AP (the only weak-supervision
signal available), but the *reported* metric is always the snippet-level one.

**Leakage investigated — and refuted.** 165 sources are shared between train and test: **59.2 % of test
clips (474/800) come from a source also present in training**. Re-scoring the saved predictions on the
two subsets (`scripts/eval_source_disjoint.py`) gives:

| subset | clips | prevalence | global pr_auc | per-video pr_auc |
|---|---|---|---|---|
| all | 800 | 0.259 | 0.8100 | 0.7492 |
| source seen in train | 474 | 0.219 | 0.7446 | 0.7020 |
| **source unseen** | 326 | 0.300 | **0.8640** | **0.8195** |

So source overlap does **not** inflate our test AP — the unseen subset scores *higher*, because it
contains a higher share of anomalous snippets (30 % vs 22 %) and less "mostly-normal film footage with
one brief event". The official split is nonetheless clip-level random, which is why the benchmark's
absolute numbers carry this optimism; we now report the source-disjoint split as a secondary,
stricter view.

**Validation split rebuilt.** `make_val` now defaults to `split_mode="group"`: all clips of one source
(film / YouTube id, everything before `__#`) stay in the same split, taken category-stratified.
`split_info.json` records train 3 516 / val 434 clips, 1 476 vs 153 sources, **source overlap = 0**.
A re-train with this stricter split is running (`runs/2026-09-23_p1_visual_swin_group`) to check that
model selection is stable; the old run is kept for the split-mode ablation.

**Next:** finish the group-split run → P2 (audio features: VGGish 128-d primary, PANNs as the event
tagger for explanations).

---

## 2026-09-23 — Session 6 — Split ablation done, audio pipeline launched

**Split-mode ablation (model selection robustness).** The source-disjoint rerun finished; test numbers
are essentially unchanged, so the reported AP is not an artifact of how validation was carved:

| metric | random split | group split (source-disjoint) |
|---|---|---|
| global `pr_auc` | 0.8100 | **0.8133** |
| global `average_precision` | 0.8099 | 0.8133 |
| per-video `pr_auc` | 0.7492 | 0.7420 |
| per-video `average_precision` | 0.7709 | 0.7642 |
| mean best IoU | 0.4328 | 0.4276 |
| macro per-class AP | 0.6756 | 0.6695 |

Also noted: the **clip-level** validation AP saturates at ~0.986 in both runs (epoch 1 already 0.96),
so weak-label validation cannot discriminate model quality beyond a coarse level. Consequence for P3:
every architecture variant is compared with an **identical training budget** on the test set, and the
clip-level validation signal is reported only as a sanity check.

**Audio backend decided (measured).** VGGish works — 10 clips extracted, `(P,128)` in [0,255], patch
counts match duration exactly (250 patches = 240 s / 0.96 s) — and loudness already separates classes
(mean patch dB: **−35.6** for a normal clip vs **−17.4** explosion, **−17.7** riot). But VGGish costs
~6.8 GFLOP per 0.96 s patch ⇒ measured ~4 patches/s ⇒ **≈50 h for all 217 h of audio** on this CPU.
Decision, recorded as a documented deviation:
- **`mel` backend is the default** — 132-d per patch (64 mel means + 64 stds + flux/centroid/flatness/zcr),
  ~1 % of VGGish's cost, ~19 s for 10 clips, and *aligned with the Edge target* (VGGish at 17 M params
  cannot run in real time on a mini-PC anyway);
- `--backend vggish` stays available for a subset ablation or a future GPU burst.

**Created:** `safewatch/features`-style extraction in `scripts/extract_audio_features.py`
(`mel`|`vggish`, per-clip quality stats: rms, rms_db, flatness, zcr, clipping ratio),
`scripts/run_p2_audio.sh` (resumable: test videos kept for the demo/UI, train chunks
download → extract → delete to bound disk), `scripts/eval_source_disjoint.py`.

**Running:** P2 data pipeline (800 test videos ≈14 GB, then 5 train chunks ≈71 GB in sequence).
Feature volume is tiny (~1.5 GB total), so disk stays comfortable (196 GB free).

**Next:** P2 dataset (pool audio patches onto the 64-frame snippet grid) → audio-only baseline →
early/late fusion comparison.

---

## 2026-09-24 — Session 7 — Audio features (partial) + the four fusion variants trained

> Written up retroactively on 2026-09-25 from `runs/`, `data/logs/` and `data/lists_partial/` — the
> session itself produced artefacts but no journal entry (the drift that Session 8 fixes).

**Audio pipeline (`scripts/run_p2_audio.sh`, resumable by design).**
- Test videos: 800 downloaded (~14 GB) and **kept** for the demo/UI; **799** mel features extracted.
  One file (`Before.Sunset.2004__#01-14-30_01-16-51_label_A.mp4`) fails the mirror's consistency check
  and is recorded in `data/features/skipped_data_video_test_videos_.json`.
- Train: chunk `1-1004` (1004 clips) and chunk `1005-2004` (996 clips) downloaded → extracted →
  videos deleted to bound disk; chunk `2005-2804` reached **100/800 downloaded** when the machine was
  shut down. At that point **2 799 / ~4 749** mel files existed (2 of 5 train chunks).
- `data/features/audio_stats.json` (per-dim mean/std used to normalise audio) was computed on
  **896 clips / 36 084 snippets** — the subset available at the time, not the whole corpus.

**Partial lists + audio-only baseline.** `data/lists_partial` = the clips that have audio
(train 1746 / val 254 / test 799, built from `data/lists`). Baseline `runs/2026-09-24_p2_audio_partial`:

| protocol | value | reference |
|---|---|---|
| val AP (clip-level, weak labels) | 0.9830 | `val_n=108` — an *earlier* val carving, not the 254-clip one |
| test global `pr_auc` | **0.3337** | prevalence 0.2588 ⇒ barely above chance when scores are pooled |
| test per-video `pr_auc` / AP | **0.6850 / 0.7090** | the same clips, ranked *within* each clip |
| mean best IoU / F1@IoU0.5 / delay | 0.382 / 0.159 / 14.0 s | |
| macro per-class clip AP | 0.154 | car_accident 0.126, explosion 0.171, abuse 0.014 |

Reading: audio discriminates well *inside* a clip but its scores are not comparable *between* clips
(global protocol ≈ chance). That is a calibration issue, not a capacity issue; per-clip normalisation
is the identified fix.

**P3 fusion: four variants implemented and trained** (`safewatch/models/fusion.py`, one flag each),
all on `data/lists_partial` (train 1746 / val 254), 30 epochs, `modality_dropout=0`:

| run | val AP (clip-level) | best epoch |
|---|---|---|
| `2026-09-24_p3_early` | 0.9926 | 28 |
| `2026-09-24_p3_late` | 0.9896 | 18 |
| `2026-09-24_p3_cross` | 0.9883 | 6 |
| `2026-09-24_p3_adaptive` | 0.9918 | 9 |

Two findings recorded as problems, not results: (a) the clip-level validation AP is saturated
(≈0.99 from epoch 1), so it **cannot rank the variants** — the spread above is noise; (b) the scorer
could not rebuild a `FusionModel` at all, so **none of the four runs had test-set numbers**. Both are
addressed in Session 8.


---

## 2026-09-25 — Session 8 — Official I3D features, the KeyError, and the fixes

**Official features and lists.** The official I3D RGB features were unzipped outside the repo
(`~/xd_official/i3d-features`, 45 GB: `RGB` 19 770 files = 3 954 clips × 5 crops, `RGBTest` 4 000 =
800 × 5, plus Flow/FlowTest). No log was kept for the unzip (`data/logs/unzip_i3d.log` is empty) — a
gap in the record, noted rather than glossed over. `data/lists_official` was built
(train 3 804 / val 150 / test 800; the test clip set is **identical** to the mirror one). The official
grid is **16 frames/snippet**, not 64 (same clip: `T=360` official vs `T=91` on the mirror).

**The failure.** The first official training attempt (10:50) died on its first batch:

```
KeyError: 'visual'   # safewatch/train.py:159 -> batch[cfg.feature_key]
```

Cause: `_make_dataset` served *visual* samples from a plain `ClipDataset`, whose item key is
`"features"`, while the training loop indexes the batch with `cfg.feature_key` (`"visual"`). The
contract had drifted when `modality` support was added for the audio run, and nothing tested it.
Traceback kept in `data/logs/p1_official_crash_keyerror.log`.

**Fixes.**
1. `safewatch/train.py` — every modality now goes through `MultimodalClipDataset`, so the batch keys
   *are* `cfg.feature_key`; `modality="both"` is rejected with a pointer to `train_fusion`;
   `TrainConfig.to_dict()` writes `in_dim` and `feature_key` into `config.json` **and** the checkpoint
   (both were properties, so a bare `asdict` lost them). Behaviour of the existing visual runs is
   unchanged bit-for-bit: the multimodal class draws the same single crop index per sample from the
   same seeded RNG and pads/masks identically, so the P1 swin results stay reproducible.
2. `safewatch/eval/runner.py` — input width is recovered from the weights (config → weights → name
   table), so `i3d_official` (1024-d) scores instead of being rebuilt as 768-d; **fusion checkpoints
   can now be scored** (kind detected from `cfg.fusion_mode`); the test list defaults to
   `<lists_dir>/test.csv`; the frame stride is inferred from the checkpoint's feature set
   (`swin_rgb` 64, `i3d_official` 16) and the scorer refuses to guess an unknown grid.
3. `safewatch/data/multimodal.py` — `AUDIO_DIM` is one constant (the zero-padding path had a
   hardcoded 132), imported by `train.py`/`runner.py`.
4. `tests/test_pipeline.py` (new, 16 tests) — dataset key == `feature_key` per modality, one shared
   crop window for visual+audio, `pool_audio` grid bands and padding, the four fusion modes' output
   contract (attention zeroed on padding, `alpha` only for adaptive, per-modality curves only for
   late), `in_dim` inference, official/fusion checkpoint rebuilding, stride resolution, and the
   split-level metric aggregation (a category with no positive clip must give NaN, not a warning).
   Suite: **29 tests, `ruff` clean**.

**First fusion test-set numbers** (all four runs scored with the fixed runner,
`data/logs/p3_score.log`; 799 test clips that have audio, mirror 64-frame grid):

| run | AP global | AP per-clip | mean best IoU | F1@IoU0.5 | delay (s) | macro per-class AP |
|---|---|---|---|---|---|---|
| early | 0.7460 | 0.7429 | 0.394 | 0.269 | 8.8 | 0.408 |
| late | **0.7511** | **0.7438** | 0.407 | 0.284 | 9.6 | 0.425 |
| cross | 0.7083 | 0.6837 | 0.397 | 0.234 | −3.1 | 0.334 |
| adaptive | 0.7321 | 0.7141 | 0.401 | 0.284 | 12.0 | 0.405 |

Read on the partial corpus: **late fusion is the best of the four** on both AP protocols, the
cross-attention variants are *worse* than plain early/late fusion, and `adaptive` buys nothing over
`late` here. That is a negative result for the P3 novelty claim, obtained on a 45 % train split — it is
a reason to re-measure after the audio corpus is complete, not a conclusion yet.

Caveats that must stay attached to that table: it is the 799-clip audio subset (not the 800-clip test
set), the train split is the partial 1746 clips, and the validation carving differs from the one the
audio-only run used (254 vs 108 clips). Fusion-vs-unimodal comparisons only become valid once the
audio corpus is complete. Note also the negative delay for `cross`: it fires *before* the annotated
event, i.e. it is early, not late — the delay is a mean over matched segments and must be read
together with the IoU.

**Jobs (re)started** — all resumable/idempotent:
- `bash scripts/run_p2_audio.sh` (appends to `data/logs/p2_audio.log`): continues chunk `2005-2804`.
  The first restart (11:48) exposed a real flaw in that script: because the train videos are deleted
  after extraction, the loop re-downloaded chunk `1-1004` (28 files, 380 MB) whose features were
  already complete, and a partially downloaded chunk would have been extracted and then deleted,
  losing the remainder. Fixed before the second restart: the script now skips a chunk when every clip
  of that chunk already has a feature file (using the mirror's own chunked `swin_rgb` dirs as the
  reference list of chunk membership) and always completes the download of a partial chunk first.
  Second restart at 11:52 went straight to `2005-2804` ("chunk 1-1004: features already complete
  (1004/1004) -> skip").
- `bash scripts/run_p1_official.sh` (new): 30-epoch visual baseline on the official features + scoring
  with the inferred 16-frame grid. Cost measured on this CPU: **48–182 s per epoch** for 3 804 clips
  (the spread is contention with the parallel download job, not a code change).
- `bash scripts/run_p3_score.sh` (new, `WAIT_FOR_TRAIN=1` guard): scores the four fusion checkpoints and
  prints the comparison table.
- `bash scripts/status.sh` now reports the three jobs plus a per-run val/test summary;
  `bash scripts/snapshot.sh` (new) tars code, docs, lists and annotations into
  `~/safewatch_snapshots/` — the safety net for the deliberate absence of Git.

**Lesson kept.** The bug that cost a run was not a modelling error but a *contract* between two
modules; the 13 metric tests could not see it. Contracts now have tests, and the run records carry
`in_dim`/`feature_key` so scoring no longer guesses.

**P1-official result (run finished 12:25).** 800 test clips, 145 649 snippets on the 16-frame grid,
prevalence 0.2379, best val AP 0.9653 at epoch 30 (still rising when the budget ran out):

| features | grid | AP global | AP per-clip | best IoU | F1@IoU0.5 | delay |
|---|---|---|---|---|---|---|
| VideoSwin, HF mirror | 64 f = 2.67 s | 0.8133 | 0.7420 | 0.428 | 0.322 | 11.8 s |
| **I3D official RGB** | 16 f = 0.67 s | **0.7223** | **0.7063** | 0.212 | 0.196 | 33.9 s |

Per class (official): car_accident 0.866, fighting 0.738, riot 0.690, explosion 0.504, shooting 0.288,
abuse 0.148.

The official 1024-d I3D features are therefore **not** automatically better than the mirror's VideoSwin
features on this pipeline — the opposite, by ~0.04 per-clip AP and much more on localisation. Keep the
comparison honest: (a) the AP floors differ (prevalence 0.2586 vs 0.2379), so only the per-clip numbers
are safely comparable, and the gap there is real; (b) I3D is a 2017-vintage backbone next to VideoSwin,
so a weaker visual prior is the first explanation; (c) the run is **under-tuned for the finer grid** —
`max_snippets=200` now covers only ~133 s of a 240-360 s clip, `min_length=2` now means 1.3 s instead of
5.3 s (hence 1 865 predicted segments against 1 238 ground-truth ones), and the validation AP was still
climbing at the last epoch. So: the AP numbers are quotable today, the localisation numbers need a
re-tune first (snippet budget covering the whole clip, segment rule in seconds, more epochs).

**Job status at write-up:** P2 audio still downloading chunk `2005-2804` (594/800 videos on disk,
2 799 features extracted), 88 GB free on disk.




---

## 2026-09-28 — Session 9 — Reliability-aware gate, robustness scoring, source-disjoint audit

**Goal.** Close the three gaps that blocked the P3/P8 claims: (a) the "adaptive" gate was trained with
`n_reliability=0`, i.e. blind to the signal-quality statistics we had already extracted; (b) removal of
a modality was a training flag with no evaluation attached; (c) all segment rules were expressed in
snippets, which means a different duration per feature grid. Plus one cheap audit that had been
available since Session 8 but never run: the source-disjoint re-scoring.

**Added — `safewatch/data/reliability.py` (7 channels, one source of truth).**
`[rms, rms_db, flatness, zcr, clip_ratio | feature_energy, temporal_motion]`. The first five come from
`data/features/audio_quality/<clip_id>.npy` (already on disk, never used); the last two are *proxies*
computed from the frozen visual features, because the benchmark track never sees pixels: descriptor
RMS activation (dark / blurred / occluded windows produce low-norm embeddings) and snippet-to-snippet
difference RMS (static or motion-blurred windows barely change). True brightness/blur need the decoded
frames and stay in the edge track - stated in the module docstring rather than implied.
`scripts/quality_stats.py` computes train-split mean/std over 3 516 clips / 231 509 snippets
(`data/features/quality_stats.json`, 20 s) and the dataset standardises with it, so no test statistics
reach the model. Measured channel spreads: `rms_db -25.1 ± 12.5`, `flatness 0.053 ± 0.117`,
`zcr 0.117 ± 0.063`, `clip_ratio 0.0017 ± 0.0166`, `feature_energy 0.305 ± 0.051`,
`temporal_motion 0.159 ± 0.074`.

**Degradation is one function, used on both sides.** `drop_modality()` zeroes a modality's features
**and** writes the sentinel `-5.0` on its reliability channels; training dropout
(`--modality-dropout 0.15`, now the default) and scoring (`--drop-modality {audio,visual}`) call the
same function, so the gate meets at test time exactly the pattern it was trained on.

**Fixed defects (all found while wiring the above).**
1. `scripts/audio_stats.py` could not run at all: `python scripts/x.py` puts `scripts/` on `sys.path`,
   not the repo root, so `import safewatch` failed - the documented invocation in its own docstring.
   Same bootstrap added to `quality_stats.py`; `eval_source_disjoint.py` already had it.
2. `AdaptiveGate` silently ignored a reliability vector when `n_reliability=0`; now it *raises* when
   channels are expected and *ignores* them when it was trained without (old checkpoints still score).
3. `standardise()` divided by `std≈0` for channels that never vary in training (`clip_ratio` on the
   mirror has std 0 in a 40-clip sample): such channels are passed through instead of amplified.
4. Segment rules in seconds: `metrics.snippets_for_seconds()` + runner `--min-seconds` /
   `--max-gap-seconds`, and `segment_rule` in `test_metrics.json` now records both units.
   2 s is 1 snippet on the 64-frame grid but 3 on the official 16-frame grid - the reason the official
   run produced 1 865 segments for 1 238 events.

**Source-disjoint audit (the number that was missing; `scripts/eval_source_disjoint.py`, no re-training).**
59 % of the mirror test clips come from a source that also appears in training, so the docstring's
hypothesis was that AP partly measures source recognition. Measured, the opposite holds - the model
scores **better** on clips whose source was never seen:

| run | features | AP global (all) | source **seen** | source **unseen** |
|---|---|---|---|---|
| `2026-09-23_p1_visual_swin` | VideoSwin 768-d, 64-frame | 0.8099 | 0.7445 (n=474) | **0.8639** (n=326) |
| `2026-09-25_p1_official_rgb` | official I3D 1024-d, 16-frame | 0.7223 | 0.6717 (n=533) | **0.7869** (n=267) |
| `2026-09-24_p3_late` (partial corpus) | swin + mel, late fusion | 0.7511 | 0.7265 (n=473) | **0.7905** (n=326) |

Reading: the headline AP is **not** inflated by source memorisation - if anything the aggregate is
conservative, because the 59 % source-shared subset is the harder one (interpretation caveat: the two
subsets also differ in class mix, so this compares difficulty of the subsets, not only source overlap).
Recorded per run as `test_metrics_source_disjoint.json`.

**Robustness pilot (before the full ladder finishes).** `runs/2026-09-24_p3_late` re-scored with the
audio branch removed drops from AP global 0.7511 to **0.7463** - late fusion is nearly insensitive to
losing audio, because the visual branch already carries most of the signal on this corpus. That is the
baseline the new adaptive gate has to beat.

**Launched: the full-corpus ladder** (`scripts/run_p3_full.sh`, resumable, log `data/logs/p3_full.log`,
started 22:33). On `data/lists` (3 516 train / 434 val / 800 test) and the complete audio corpus:
audio-only -> early -> late -> cross -> adaptive (reliability gate, modality dropout 0.15), each scored
three ways (both / `--drop-modality audio` / `--drop-modality visual`). First epoch of the audio-only
rung: val AP 0.7647 in 65 s/epoch (~33 min per model). `scripts/fusion_table.py` prints the whole
ladder including the two degradation columns.

**Quality gates.** 36 tests green (29 -> 36: reliability layout/standardisation, dataset alignment,
degradation sentinels, gate sensitivity, seconds conversion, end-to-end `--drop-modality` scoring),
`ruff` clean.

**Next.** Read the full ladder when it lands and state plainly whether fusion beats the visual-only
0.8099 / 0.7642 baseline on the same corpus - if it still does not, the audio score calibration
(per-clip normalisation) is the first suspect, not the fusion architecture.



---

## 2026-09-28 (suite) — Session 9b — Premier étage du ladder complet + le défaut de calibration

**Premiers résultats sur le corpus complet** (`data/lists`, 3 516 train / 434 val / 800 test, audio
complet). Comparaison désormais légitime : même liste d'entraînement que la référence visuelle.

| run | AP global | AP per-clip | IoU moy. | F1@0.5 | AP global, audio retiré | vidéo retirée |
|---|---|---|---|---|---|---|
| `2026-09-23_p1_visual_swin` (référence) | 0.8099 | 0.7709 | 0.433 | 0.322 | — | — |
| `2026-09-28_p2_audio_full` | 0.4464 | 0.6252 | 0.302 | 0.183 | **0.2463** | — |
| `2026-09-28_p3full_early` | 0.8015 | **0.7818** | 0.416 | 0.310 | **0.8138** | 0.4621 |
| `2026-09-24_p3_*` (corpus partiel, 45 %) | 0.708-0.751 | 0.684-0.766 | ~0.40 | ~0.28 | — | — |

Deux lectures, une bonne et une mauvaise :

1. **Bonne** : la fusion précoce dépasse enfin la référence visuelle sur l'AP per-clip (0.7818 contre
   0.7709) et le corpus complet a fait remonter l'audio seul de 0.334 à 0.446 en AP global. La
   localisation reste du même ordre (IoU 0.416, F1@0.5 0.310).
2. **Mauvaise, et c'est la vraie information** : **retirer l'audio améliore** la fusion précoce
   (AP global 0.8015 -> 0.8138) ; retirer la vidéo l'effondre (0.8138 -> 0.4621, soit le niveau de
   l'audio seul). Le modèle n'utilise donc pas l'audio comme un apport mais le subit comme un bruit
   dans le protocole global. Même diagnostic que Session 7, maintenant chiffré : les niveaux de score
   audio ne sont pas comparables **entre** clips (AP global 0.446 contre per-clip 0.625), donc le
   pooling inter-clips détruit la courbe PR.

**Le candidat de correction est implémenté** : `MultimodalClipDataset(audio_norm="clip")`
(instance-normalisation par clip, invariante à un gain/offset près - `clip_normalise()`), exposé en
config (`--audio-norm clip`) sur `safewatch.train` et `safewatch.train_fusion`, avec tests.
L'expérience est lancée en file d'attente : audio seul + clipnorm, late + clipnorm, adaptive + clipnorm
(donc le portail de fiabilité ET la calibration dans la même configuration), chacune re-scorée avec et
sans chaque modalité. Scripts : `scripts/run_p3_clipnorm.sh`, file d'attente `scripts/run_p3_catchup.sh`.

**Incident à consigner (leçon utile).** Le premier entraînement `late` du ladder est mort avec
`AttributeError: 'MultimodalClipDataset' object has no attribute 'audio_norm'`. Cause : le paquet a été
édité **pendant** que la tâche tournait. Python 3.14 utilise `forkserver` par défaut pour
`multiprocessing`, donc un worker de DataLoader **ré-importe** `safewatch.data.multimodal` depuis le
disque : il a exécuté le nouveau `_audio()` sur un objet dataset sérialisé par l'ancien `__init__`.
Aucune donnée abîmée, seul l'étage `late` manquait ; il est rejoué par le script de rattrapage.
Règle retenue : **on ne modifie pas le paquet pendant qu'une tâche tourne, on met la modification en
file d'attente** - ce qui est exactement ce que fait `run_p3_catchup.sh`.


---

## 2026-09-28 (suite) — Session 9c — P7 Edge : export ONNX, int8, budget mesuré

**Nouveau : `scripts/export_onnx.py`** (P7, jusqu'ici 0 %). Il exporte un checkpoint en ONNX (axe
temps dynamique), quantifie en int8, et écrit `export/onnx/<run>_{fp32,int8}.onnx` +
`<run>_report.json` (tailles, latences p50/p95, parité, charge machine).

**Résultats mesurés** (CPU, 3 threads, `loadavg_1m ≈ 16` : la mesure a été prise pendant que le ladder
tournait, donc les latences sont **majorées** ; à re-mesurer au repos et à citer avec cette réserve) :

| modèle | fp32 ONNX | parité vs PyTorch | p50 / 100 snippets | int8 | int8 p50 | int8 parité logits |
|---|---|---|---|---|---|---|
| `p3full_early` (fusion) | 2.331 MB | **7.6e-06** | 5.70 ms (17.5k snip/s) | 0.597 MB (×0.26) | 12.52 ms (×0.46) | 0.157 ❌ |
| `p1_visual_swin` (MIL) | 2.073 MB | **9.5e-06** | 3.42 ms (29.2k snip/s) | 0.533 MB (×0.26) | 11.69 ms (×0.29) | 0.146 ❌ |

Trois constats, dont deux négatifs - donc utiles :

1. **La tête de décision n'est pas le goulot d'étranglement.** Une fenêtre de 100 snippets couvre ≈4.4 min
   de vidéo (2.67 s/snippet) et coûte ~6 ms même sur une machine chargée ⇒ le budget temps réel se joue
   dans l'**extraction de features**, pas dans la fusion. C'est l'argument qui justifie la piste
   « extracteur léger ONNX int8 » de la piste Edge, et il fallait le mesurer.
2. **La quantification dynamique int8 ne fait pas gagner de temps ici** : le graphe rétrécit de 4× mais
   ralentit de 2 à 3.4× (quantification/déquantification sur des MatMul minuscules). Résultat négatif
   assumé : int8 est un gain de *taille*, pas de *latence*, pour un modèle de cette taille.
3. **int8 déplace les logits de ~0.15** (> tolérance 0.05 fixée dans le script) ⇒ refusé pour le
   déploiement en l'état ; il faudrait une recalibration des seuils (P5) avant de l'utiliser. Le script
   refuse et le dit au lieu d'écrire un fichier « ok ».

**Piège d'outillage documenté (coûteux à trouver).** Le premier rapport affichait une parité de **10.36**
alors que le graphe était correct. Cause : `torch.onnx.export` (exporteur TorchScript historique)
**mute le modèle en place** — le même modèle renvoie des logits différents *après* l'export (mesuré
`before-vs-after = 7.91`, `before-vs-onnx = 6.7e-06`). Le script capture donc désormais sa référence
**avant** d'exporter, et le rapport le mentionne (`pytorch_reference_taken`). Détail : la mutation ne se
reproduit pas sur le petit modèle de test, la garde de parité du test unitaire porte donc sur le graphe
(ce qui compte) et non sur la mutation.
Autre piège corrigé : l'exporteur **élague** les entrées non lues (early/late/cross n'utilisent pas le
vecteur de fiabilité) - nourrir 4 entrées en dur fait échouer ONNX Runtime ; les feeds sont maintenant
construits depuis `session.get_inputs()`.

**Reste à faire côté Edge** : extracteur léger (ONNX int8) pour la piste démo depuis les pixels/audio
bruts, export Jetson/TensorRT (à simuler, pas de matériel ici), et re-mesure du tableau ci-dessus au
repos.


---

## 2026-09-29 — Session 9d — Coupure machine : ce qui est mort, ce qui a survécu, comment on reprend

**Ce qui s'est passé.** La machine a été éteinte le 2026-09-28 vers **23:36** (redémarrage le 29 à
07:09, `uptime` le confirme). Tout ce qui tournait est mort **sans message d'erreur** : le log s'arrête
simplement. C'est le mode de défaillance le plus dangereux de ce projet (pas de Git, tâches longues),
donc il est consigné comme tel.

**État à la reprise (vérifié, pas supposé)** :

| run | état | conséquence |
|---|---|---|
| `2026-09-28_p2_audio_full` | 30 epochs + checkpoint + 2 fichiers de métriques | terminé |
| `2026-09-28_p3full_early` | 30 epochs + checkpoint + **3** fichiers de métriques | terminé (les deux ablations de modalité incluses) |
| `2026-09-28_p3full_cross` | **23/30 epochs**, checkpoint de l'epoch 16, pas de `metrics.json` | conservé sous `..._p3full_cross_interrupted_ep23/` et **retrainé de zéro** : tous les étages du ladder doivent avoir le même budget (30 epochs), sinon le tableau compare des budgets différents |
| `2026-09-28_p3full_late`, `..._adaptive` | jamais démarrés | relancés par la file |

**Reprise.** `scripts/run_p3_catchup.sh` relancé : il rejoue le ladder (les runs terminés sont sautés,
`late`/`cross`/`adaptive` sont entraînés puis re-scorés) puis enchaîne sur l'expérience de calibration.
Nouveau garde-fou : **`scripts/resume_jobs.sh`** — si aucun job du projet ne tourne, il relance la file,
sinon il refuse et affiche ce qui tourne. C'est la commande à lancer après *n'importe quelle* coupure
(elle peut être mise dans un `@reboot` de cron).

**Leçon de conception, pas de malchance.** La résilience de ce repo ne vient pas des sauvegardes mais du
fait que chaque script est **idempotent et repris-en-cours** : `run_p2_audio.sh` (les chunks complets
sont sautés), `run_p3_full.sh` (un étage avec checkpoint est sauté), `run_p3_clipnorm.sh`,
`run_p3_catchup.sh`. La coupure n'a donc coûté que les epochs en cours (≈23 epochs de `cross`), pas les
≈3 h de travail déjà validé. Corollaire : un run **sans** `metrics.json` doit être considéré comme
incomplet, même s'il a un checkpoint - c'est exactement le piège que masquait l'ancienne logique de saut.


---

## 2026-09-29 — Session 9e — Ladder complet : late gagne, cross/adaptive perdent, la calibration échoue

**Le tableau complet** (corpus complet, 3 516 entraînement / 434 validation / 800 test, 30 epochs pour
chaque étage, mêmes features, même grille, chaque modèle re-scoré avec la modalité retirée) :

| run | AP glob | AP per-clip | IoU moy. | F1@0.5 | −audio | −vidéo |
|---|---|---|---|---|---|---|
| `p1_visual_swin` (**référence visuelle**) | 0.8100 | 0.7709 | **0.433** | **0.322** | — | — |
| `p3full_early` | 0.8015 | 0.7818 | 0.416 | 0.310 | 0.8138 | 0.4621 |
| **`p3full_late`** | **0.8125** | **0.7825** | 0.423 | 0.306 | 0.8133 | 0.3769 |
| `p3full_cross` | 0.7981 | 0.7613 | 0.402 | 0.208 | 0.7946 | 0.4281 |
| `p3full_adaptive` (portail 7 canaux + dropout) | 0.7553 | 0.7462 | 0.382 | 0.211 | 0.7731 | 0.4185 |
| `p3full_late_clipnorm` | 0.8135 | 0.7703 | 0.426 | **0.314** | 0.8172 | 0.2637 |
| `p3full_adaptive_clipnorm` | 0.7668 | 0.7520 | 0.422 | 0.210 | 0.7788 | 0.3062 |
| `p2_audio_full` (audio seul) | 0.4464 | 0.6252 | 0.302 | 0.183 | 0.2463 | — |
| `p2_audio_full_clipnorm` | 0.3285 | 0.5931 | 0.261 | 0.134 | 0.6034 ⚠ | — |

**Ce qui est démontré.** La fusion **late** dépasse la référence visuelle sur les deux protocoles
(+0.0025 en AP global, +0.0116 en AP per-clip) à budget d'entraînement identique : le cœur de la
promesse du sujet est donc tenu - **mais de peu en global**, et la localisation reste légèrement
meilleure en visuel seul (IoU 0.423 contre 0.433, F1@0.5 0.306 contre 0.322). À dire tel quel.

**Ce qui est réfuté, et c'est le résultat le plus utile.** Les deux variantes « sophistiquées » -
attention croisée (`cross`, 0.7981 / 0.7613) et surtout le portail de fiabilité (`adaptive`, 0.7553 /
0.7462 ; avec clipnorm 0.7668 / 0.7520) - sont **nettes derrière** l'early et la late. C'est la
deuxième fois (cf. Session 7 sur corpus partiel) et cette fois avec le corpus complet, le portail
alimenté par les 7 canaux et le dropout de modalité : la piste « cross-attention + gating » **ne paie
pas** sur ces features ; la fusion simple gagne. Conclusion pour le rapport : ne pas vendre l'attention
croisée comme la contribution ; la contribution mesurable est (a) la chaîne multimodale faiblement
supervisée complète, (b) les métriques et le protocole, (c) la robustesse quantifiée.

**L'hypothèse de calibration est réfutée elle aussi** (expérience `--audio-norm clip`, normalisation
d'instance par clip) :

- audio seul : AP global **0.446 -> 0.329**, per-clip 0.625 -> 0.593 ⇒ la normalisation **détruit** de
  l'information (le niveau sonore absolu - une explosion est *forte* - est discriminant) ;
- late + clipnorm : AP global 0.8125 -> **0.8135** (bruit), mais per-clip 0.7825 -> **0.7703** (perte) ;
  en échange, la localisation s'améliore un peu (F1@0.5 0.306 -> 0.314) et le modèle dépend *davantage*
  de l'audio (−vidéo : 0.3769 -> 0.2637).
- Et **retirer l'audio continue d'améliorer ou de ne pas dégrader** le score global (0.8135 -> 0.8172)
  ⇒ le défaut n'était pas le gain/offset par clip.

**Piège de lecture évité de justesse.** Dans `p2_audio_full_clipnorm`, retirer la modalité donne un AP
global de **0.6034**, ce qui ressemble à « mieux sans audio » : c'est en réalité l'**artefact de
dégénérescence déjà documenté** (`pr_auc` trapézoïdal d'un score constant = `(prévalence+1)/2` = 0.629
ici). Entrée constante ⇒ toutes les fenêtres obtiennent le même logit ⇒ le classement inter-clips
n'existe plus. C'est exactement pourquoi les deux protocoles AP sont toujours imprimés côte à côte, et
pourquoi l'AP per-clip du même run reste à 0.5265 (≈ hasard intra-clip). **Ne jamais citer la colonne
« −modalité » sans l'artefact.**

**Où est réellement le verrou audio ?** Les features audio sont des statistiques de patch log-mel
(132-d) extraites à la main ; le VGGish 128-d (celui du papier XD-Violence) n'existe que pour **10
clips** sur 4 749 faute de budget CPU. La conclusion honnête est donc : *la fusion multimodale est
implémentée et mesurée, mais l'apport audio est plafonné par la représentation audio, pas par
l'architecture de fusion*. Prochains candidats, par ordre de valeur : (1) VGGish sur le corpus complet
(extraction ~2-4 h CPU), (2) calibration **au niveau décision** (température/isotonic par clip, P5)
plutôt que sur les features, (3) accepter la marge de la late fusion et le dire.

## 2026-09-30 — Session 10 — Échelle 30 epochs sur la grille officielle (I3D + VGGish) : l'apport audio est enfin réel

**But.** La session 13 laissait une conclusion inconfortable : sur la grille miroir (VideoSwin + log-mel
132-d), la fusion ne battait le visuel seul que de +0.0025 en AP global, et retirer l'audio ne coûtait
rien. Le verrou identifié était la **représentation audio**, pas l'architecture. Il fallait donc la
même échelle d'ablation (30 epochs, même protocole, même liste) avec le **VGGish officiel 128-d** —
celui du papier XD-Violence — sur la grille officielle 16 frames/snippet (I3D 1024-d).

**Résultat (test 800 clips, jeu de clips identique à la grille miroir — vérifié : recouvrement 800/800).**

| run (30 ep) | APglob | APclip | IoU* | F1@.5* | délai* | −audio | −vidéo |
|---|---|---|---|---|---|---|---|
| `cross` | **0.7745** | **0.7961** | **0.369** | 0.243 | **5.4 s** | 0.7243 | 0.6563 |
| `late` | 0.7719 | 0.7857 | 0.354 | **0.265** | 24.6 s | 0.7178 | 0.6445 |
| `early` | 0.7643 | 0.7830 | 0.351 | 0.260 | 25.7 s | 0.7141 | 0.6514 |
| `adaptive` | 0.7281 | 0.7665 | 0.228 | 0.180 | 18.8 s | 0.7087 | 0.5891 |
| visuel seul | 0.7259 | 0.7148 | 0.335 | 0.225 | 25.7 s | — | — |
| audio seul | 0.6403 | 0.6986 | 0.337 | 0.201 | 27.1 s | — | — |
| *réf. miroir `p3full_late`* | 0.8125 | 0.7825 | 0.423 | 0.306 | 11.4 s | 0.8135 | 0.3769 |

\* règle de segmentation **en secondes** (min 5.333 s / gap 2.667 s), voir le piège ci-dessous.

**Ce qui est démontré, et c'est la première fois franchement.** Avec le VGGish officiel, la fusion
dépasse **sa propre** référence visuelle de **+0.0486 en AP global et +0.0813 en AP per-clip** — soit
≈20× et 7× les marges de la grille miroir. Surtout, les deux modalités deviennent **nécessaires** :
retirer l'audio coûte −0.050, retirer la vidéo −0.118. L'hypothèse « le verrou est la représentation
audio » est donc **confirmée** : ce n'était pas l'architecture de fusion, c'était le log-mel 132-d.

**Ce qu'il faut dire honnêtement en même temps.** L'AP global absolu reste **inférieur** à la grille
miroir (0.7745 contre 0.8125). Ce n'est pas la fusion qui recule, c'est le **backbone visuel** : le
visuel seul passe de 0.8100 (VideoSwin 768-d) à 0.7259 (I3D 1024-d, grille 4× plus fine). La fusion
officielle rattrape presque tout l'écart et devient le **meilleur AP per-clip du projet** (0.7961 contre
0.7825). Conclusion pour le rapport : le gain multimodal est réel mais il faut le comparer **à
backbone égal** ; la comparaison inter-grilles mélange deux facteurs.

**Piège de lecture évité (segmentation).** La règle de segments est exprimée en **snippets**
(`min_length=2`, `max_gap=1`). Sur la grille miroir (64 frames/snippet) cela vaut 5.333 s ; sur la
grille officielle (16 frames/snippet) seulement **1.333 s** — soit 4× plus fin. La première lecture
donnait donc l'officiel à IoU 0.21–0.26, délai 18–37 s et 1 356–2 058 segments, ce qui ressemble à une
localisation dégradée alors que c'est un **artefact de règle**. Re-scoré à règle wall-clock égale
(`--min-seconds 5.333 --max-gap-seconds 2.667`), le nombre de segments retombe (696–957) et les
chiffres deviennent comparables. Leçon : ne **jamais** comparer des métriques de localisation entre
grilles sans normaliser la règle en secondes. Les sorties de la règle juste sont dans
`runs/*30ep*/seg_seconds/`.

**Calibration (P5, résultat négatif).** Température ajustée sur la val officielle (150 clips seulement,
taux de positifs 0.520 — donc beaucoup plus mince que la val miroir de 434 clips) :

- `late` : T=1.3601, NLL 0.1235 → **0.1161** (mieux), mais ECE **0.0348 → 0.0369** sur val et
  **0.0181 → 0.0244** sur test (pire) ;
- `cross` : T=1.7035, NLL 0.1544 → **0.1261** (mieux), ECE 0.0421 → 0.0427 sur val, 0.0313 → 0.0311
  sur test (plat).

Autrement dit : les scores bruts sont **déjà bien calibrés** (ECE test 0.018 pour `late`) et la
température n'améliore que le NLL, pas le calibrage. Ajuster T sur 150 clips abîme même le test pour
`late`. À ne pas vendre comme « calibration validée » : c'est une mesure négative utile, et le candidat
suivant est la calibration **par classe** (le tableau per-classe montre que les erreurs sont
concentrées).

**Ce qui reste cassé.** `abuse` plafonne à **0.138** (meilleur cas, `late`) — soit 11 clips de test,
donc statistiquement fragile, mais ce n'est pas publiable en l'état. L'AP par classe ailleurs est solide
(riot 0.932, car_accident 0.924, fighting 0.900, explosion 0.745 en `late`) ; `shooting` reste moyen
(0.681). L'`adaptive` (portail de fiabilité) est **confirmé une troisième fois** comme la pire variante
(0.7281 ; −vidéo 0.5891) — la piste gating est à abandonner explicitement dans le rapport.

**Artefacts.** 6 runs (`runs/2026-09-29_p2vggish30ep_*`, `runs/2026-09-29_p3vggish30ep_*`,
`runs/2026-09-30_p3vggish30ep_adaptive`, `..._early`) avec `test_metrics.json`,
`predictions*.npz`, `calibration.json`, `seg_seconds/` ; journal complet
`data/logs/p3_vggish_30ep.log` (01:28, ~2h40 pour toute l'échelle) ; script reproductible
`scripts/run_p3_vggish.sh` (`EPOCHS=30 SUFFIX=30ep`).

### Addendum — audit Tier 1 (mêmes runs, aucun ré-entraînement)

Quatre vérifications demandées avant de citer le résultat ci-dessus.

**1. Fuite de source : le gain multimodal y survit.** 66.6 % des 800 clips de test partagent leur
source (film / vidéo YouTube) avec l'entraînement. Sur le sous-ensemble **source-inédite** (267 clips),
tout va **mieux** que sur les 533 clips vus ⇒ pas d'inflation :

| run | APglob (vus) | APglob (inédits) | APclip (vus) | APclip (inédits) |
|---|---|---|---|---|
| `cross` | 0.7435 | **0.8452** | 0.7817 | **0.8258** |
| `late` | 0.7380 | 0.8309 | 0.7688 | 0.8204 |
| visuel seul | 0.6733 | 0.7980 | 0.6729 | 0.8008 |
| audio seul | 0.6528 | 0.6541 | 0.7313 | 0.6316 |

Lecture : `cross` garde **+0.047 AP global** sur le visuel seul dans le sous-ensemble sans fuite (même
ordre de grandeur qu'en global), et l'audio seul est le seul run insensible au partage de source
(0.6528 vs 0.6541) — cohérent : le son ne dépend pas de l'identité du film. Script :
`scripts/eval_source_disjoint.py`, sortie `test_metrics_source_disjoint.json`.

**2. Localisation par catégorie : deux défauts systématiques.** Les annotations d'intervalles sont
**binaires** (pas de colonne catégorie), donc la localisation par catégorie est définie comme la
restriction aux clips dont l'ensemble de labels contient la catégorie
(`scripts/per_class_localization.py`) :

| catégorie | clips | segments GT | segments prédits | IoU | F1@.5 | délai |
|---|---|---|---|---|---|---|
| fighting | 126 | 275 | 192 | 0.438 | 0.334 | −3.3 s |
| shooting | 104 | 261 | 120 | 0.297 | 0.126 | −17.1 s |
| riot | 101 | 157 | 240 | 0.410 | 0.448 | **+38.3 s** |
| abuse | 11 | 19 | 12 | 0.424 | 0.323 | −2.7 s |
| car_accident | 106 | **513** | 151 | 0.245 | **0.054** | −19.0 s |
| explosion | 103 | 192 | 127 | 0.340 | 0.245 | −12.9 s |

- **`riot` est systématiquement en retard** : +38 s (`cross`), +62 s (`late` et visuel officiels),
  +48 s (miroir). Une émeute se détecte quand elle éclate, pas quand elle commence — c'est un biais
  exploitable, mais il faut le dire dans la fiche d'incident (« détection tardive sur les émeutes »).
- **`car_accident` est sous-segmenté** : 513 événements annotés (~4.8 par clip) pour 151–234 segments
  prédits, F1@0.5 0.054–0.174. Les accidents sont courts et multiples dans un même clip ; le lissage
  temporel du MIL les fusionne en un seul segment et en rate la majorité.

**3. Diagnostic de la classe `abuse` — c'est une famine de données, pas un défaut de features.**

- entraînement : **50 clips positifs sur 3 804 (1.3 %)** contre 367–463 (9.6–12 %) pour les cinq
  autres classes ; test : 11 clips ;
- la tête de catégorie **ne se déclenche jamais** : p(abuse) moyenne 0.069 sur ses positifs contre
  0.028 sur les négatifs, **0 vrai positif et 0 faux positif au seuil 0.5** (les négatifs les mieux
  notés sont des clips `fighting` / `shooting`, donc du contenu violent proche) ;
- mais ses incidents sont **bien localisés** : tIoU **0.460**, F1@0.5 0.323 ⇒ la tête binaire
  d'incident fonctionne sur ces clips ; c'est uniquement la **classification** de catégorie qui est
  morte.

L'AP de 0.138 vient donc d'un classement résiduel (0.069 > 0.028) sur 11 clips — un chiffre à ne pas
publier comme une performance, mais comme la preuve de la famine. Script :
`scripts/class_diagnostic.py` → `class_diagnostic.json` par run.

Le même diagnostic sur `cross` confirme la famine (`abuse` : 0 TP / 0 FP, p=0.081 vs 0.030) et révèle
la contrepartie de son meilleur AP global : sa tête `shooting` est **beaucoup plus agressive**
(96 TP mais **114 faux positifs** au seuil, contre 68 pour `late`), et elle localise moins bien
(tIoU 0.297 contre 0.555). `cross` gagne donc au classement inter-clips ce qu'il perd en précision sur
`shooting` — à vérifier avant de le promouvoir comme modèle de production.

**4. La règle de segments est désormais en secondes par défaut** dans `scripts/run_p3_vggish.sh`
(`MIN_SECONDS=5.333` / `MAX_GAP_SECONDS=2.667`, toutes les ablations), le scorer **affiche** la règle
en snippets *et* en secondes au démarrage (`[score] ... rule=(8,4) snippets = (5.33,2.67) s`), et
chaque scoring produit aussi son `per_class_localization.json`. 6 tests de régression supplémentaires
(`tests/test_analysis_scripts.py`, 63 au total) verrouillent : conversion secondes→snippets par grille,
restriction conjointe prédictions+GT par catégorie, `min_length` en secondes, résolution de la grille
(sans valeur par défaut : échec bruyant) et comptage TP/FP du diagnostic de classe.

**Ce que ça change au bilan.** Les chiffres de la session restent valides (mêmes runs, aucun
ré-entraînement) et le gain multimodal est **audité** (pas de fuite). En revanche deux faiblesses de
localisation sont maintenant chiffrées et attribuées (retard `riot`, sous-segmentation
`car_accident`), et la classe faible est expliquée par sa cause réelle. La suite par valeur : (1)
`abuse` — pondération de classe / sur-échantillonnage, pas de nouvelle feature ; (2)
`car_accident` — une tête de densité ou un seuil plus bas pour les accidents multiples ; (3)
`riot` — décalage du seuil de déclenchement vers le début.


## 2026-09-30 — Session 11 — Pondération de classe et robustesse : le diagnostic était bon, le remède non

### A. La tête multi-label est-elle affamée ? (hypothèse testée, moitié réfutée)

Mise en œuvre : `pos_weight` par catégorie dans `multilabel_loss` (sémantique
`nn.BCELoss(pos_weight=...)`, appliquée à la main pour ne pas changer la forme probabiliste des
checkpoints existants ; équivalence vérifiée contre `BCEWithLogitsLoss` dans un test). Deux modes :
`auto` = `n_neg/n_pos` (poids mesurés sur la grille officielle : `abuse=75.1`, les cinq autres
7.2-9.4) et `sqrt` = amorti (`abuse=8.7`, autres 2.6-3.1). Nouveau champ `--multi-pos-weight`.

| run | APglob | APclip | macro | `abuse` | `shooting` | IoU | F1@.5 |
|---|---|---|---|---|---|---|---|
| late (référence) | 0.7719 | 0.7857 | 0.720 | 0.138 | 0.681 | 0.247 | 0.258 |
| late + `auto` | 0.7747 | 0.7871 | 0.757 | **0.096** | 0.731 | 0.357 | 0.267 |
| late + `sqrt` | 0.7753 | 0.7869 | 0.747 | **0.094** | 0.709 | 0.360 | 0.271 |
| cross (référence) | 0.7745 | 0.7961 | 0.696 | 0.081 | 0.563 | 0.260 | 0.239 |
| cross + `auto` (best) | 0.7666 | 0.7903 | **0.785** | **0.250** | 0.704 | 0.374 | 0.248 |
| cross + `auto` (**last**) | 0.7524 | 0.7852 | **0.790** | **0.259** | 0.723 | 0.358 | 0.223 |

- **Le diagnostic est confirmé causalement** : avec un poids de 75, la tête `abuse` se réveille
  (0 → 7 vrais positifs sur 11 ; p(abuse) moyenne 0.069 → 0.633). La famine de données était réelle.
- **Mais le remède est le mauvais outil pour une métrique de classement.** Sur `late`, l'AP `abuse`
  *baisse* (0.138 → 0.096) alors que le rappel au seuil monte : l'AP mesure l'ordre, et la pondération
  achète du rappel en dégradant l'ordre (146 faux positifs sur 789 négatifs). Amortir (`sqrt`, poids
  8.7) ne corrige rien (0.094) : ce n'est pas une question de dosage.
- **L'effet dépend de l'architecture de fusion** : `cross` gagne au contraire +0.169 d'AP `abuse`
  (0.081 → 0.250) et +0.089 de macro pour −0.008 d'AP global seulement, quand `late` perd de l'AP
  `abuse`. Une conclusion tirée d'un seul run aurait été fausse — à présenter comme une interaction.
- Effets de bord sur les deux runs : **toutes** les têtes sur-déclenchent (faux positifs ×2 à ×4 :
  fighting 34→83, shooting 68→121, riot 14→33, car_accident 18→76, explosion 19→65, abuse 0→146) et
  la localisation s'améliore nettement (IoU 0.247 → 0.357 pour `late`, 0.260 → 0.374 pour `cross`),
  mais c'est un effet secondaire non isolé (un seul seed) : à ne pas vendre comme reproductible.
- **Découverte structurelle** : la val officielle (150 clips) **ne contient aucun clip `abuse`**
  (car_accident 21, fighting 18, riot 17, shooting 16, explosion 10, abuse 0) ⇒ la sélection de modèle
  en était structurellement aveugle, indépendamment des 50 clips d'entraînement. D'où un avertissement
  au démarrage et la sauvegarde d'un `ckpt_last.pt` — qui paie immédiatement : sur `pwcross`, le
  dernier epoch est **meilleur** en catégorisation (abuse 0.259, macro 0.790) et **moins bon** en
  global (0.7524 contre 0.7666). Le critère binaire n'est pas le bon quand une classe est absente de
  la val.

**Conclusion pour le rapport** : la cause est la famine (50/3 804 = 1.3 %, contre 9.6-12 % pour les
autres), la pondération inverse **prouve** la cause mais ne règle pas la métrique ; il faut plus de
données `abuse`, pas un poids plus fort.



### B. Robustesse (P8) — quatre familles de dégradation, en **espace de features**

`scripts/robustness_sweep.py` + `safewatch/eval/robustness.py`, avec un point d'accroche `perturb`
**dans** `score_clips` (un seul chemin de scoring, donc pas de second scorer qui dérive). Règle
segmentaire en secondes, `--figure` pour les courbes, `scripts/robustness_table.py` pour le tableau.
**Avertissement assumé** : ce sont des perturbations de la *représentation* (features précalculées),
pas des médias. Seul le log-mel est régénérable depuis les vidéos (~3 s/clip mesuré, ≈2.5 h pour un
balayage SNR à 4 niveaux sur 800 clips) ; VGGish et I3D ne le sont pas (pas de poids/extracteur ici).

| dégradation | modèle | propre | pire point | Δ APglob | Δ APclip | Δ macro |
|---|---|---|---|---|---|---|
| audio : atténuation → silence | `cross` (VGGish) | 0.7745 | 0.7243 | **−0.050** | −0.071 | **−0.100** |
| audio : atténuation → silence | `late` (VGGish) | 0.7719 | 0.7178 | −0.054 | −0.062 | −0.142 |
| audio : atténuation → silence | `p3full_late` (**mel**) | 0.8143 | 0.8133 | **−0.001** | −0.009 | −0.005 |
| audio : bruit additif 0 dB (bruit = signal) | `cross` | 0.7745 | 0.7768 | **+0.002** | −0.011 | −0.005 |
| audio : 75 % des snippets perdus | `cross` | 0.7745 | 0.7594 | −0.015 | −0.054 | −0.003 |
| visuel : occlusion 30 s | `cross` | 0.7745 | 0.7623 | −0.012 | −0.053 | −0.072 |
| visuel : occlusion 30 s | `late` | 0.7719 | 0.7480 | −0.024 | −0.038 | −0.072 |
| visuel : occlusion 30 s | `adaptive` (portail **aveugle**) | 0.7281 | 0.7192 | −0.009 | −0.034 | −0.074 |
| visuel : occlusion 30 s | `adaptive` (portail **prévenu**) | 0.7281 | **0.7389** | **+0.011** | −0.025 | −0.076 |

- **Contrôle de cohérence gratuit et décisif** : l'atténuation niveau 0 doit reproduire la ligne
  `--drop-modality`. C'est vrai au 3e-9 d'AP près pour les trois modèles (0.7243 / 0.7178 / 0.8133).
  C'est ce contrôle qui a révélé le bug de référentiel (3 ci-dessous).
- **Ce qui coûte, c'est la suppression d'information, pas la perturbation** : à 0 dB (autant de bruit
  que de signal) l'AP global est *inchangé* (+0.002) et l'AP per-clip ne perd que 0.011 ; le silence
  audio coûte 0.050 d'AP global. Le bruit additif est moyenné par l'encodeur, l'absence ne l'est pas.
- **Le modèle mel ne perd rien quand on coupe l'audio (−0.001)** : il ne s'en servait pas. Le VGGish
  en perd 0.050 — cohérent avec tout le reste du projet : c'est la représentation qui rend l'audio
  réellement utile.
- **Le portail de fiabilité : la réplication ne confirme pas.** Sur le seed d'origine (42), prévenu que le
  flux visuel est dégradé, l'`adaptive` gagnait **+0.016 / +0.020 d'AP** face au même run aveugle
  (15 s / 30 s d'occlusion). Deux seeds supplémentaires (43, 44), même config, même principe, avec en
  plus un placement d'occlusion **différent** (seed 1), donnent : **+0.0164 / −0.0041 / +0.0314** (15 s)
  et **+0.0197 / −0.0023 / +0.0365** (30 s) — soit **2 seeds positifs sur 3**, moyenne +0.015 / +0.018,
  avec un seed strictement nul. Verdict honnête : **l'effet n'est pas établi**, il est au mieux
  suggéré.
- **Et l'argument décisif est le bruit de seed** : sur les trois seeds, le modèle `adaptive`
  **propre** donne 0.7281 / 0.6319 / 0.7099, soit une dispersion de **0.096 d'AP** (σ = 0.051) — six
  fois l'effet revendiqué. Un gain de +0.015 ne peut donc pas être distingué du bruit de seed de la
  variante à laquelle il s'applique : il faudrait beaucoup plus de seeds, pas un seed de plus. À cela
  s'ajoute le tell déjà signalé (2 seeds sur 3 font *mieux* que leur propre niveau propre, ce qui sent
  le décalage de point de fonctionnement). Conclusion pour le rapport : **ne pas présenter le portail
  comme un gain** — il reste, au mieux, « non réfuté », et il est par ailleurs la variante la plus
  instable des quatre.
- L'occlusion visuelle est mieux tolérée que la perte audio, et 75 % de snippets audio perdus ne
  coûtent que 0.015 d'AP global ⇒ redondance temporelle élevée.

### Quatre bugs à moi, tous attrapés par la mesure (la leçon vaut plus que les chiffres)

1. **Décorateur déplacé** : une insertion d'une ligne a laissé `@torch.no_grad()` sur la mauvaise
   fonction. Symptôme : `Cannot call numpy() on Tensor that requires grad` au **premier** rapport
   d'epoch — deux minutes d'entraînement perdues, `cross` tué d'entrée. `late` avait survécu
   uniquement parce que son processus était antérieur à l'édition.
2. **Accroche qui rend la dégradation dans le mauvais slot** (les deux branches d'un ternaire étaient
   identiques) : silencieux quand les deux flux ont la même largeur, erreur conv1d incompréhensible
   sinon (visuel 1024 vs audio 128). Quatre balayages morts.
3. **Référentiel mélangé** : le bruit utilisait l'écart-type de la *release* VGGish (médiane 0.222)
   alors que le modèle lit un flux **normalisé** (0.805) ⇒ la ligne « 0 dB » valait ≈ +10 dB et le
   balayage sous-estimait les dégâts d'un facteur ~3 ; et l'atténuation s'effondrait sur la moyenne
   *brute* (constante DC) au lieu du silence. Attrapé par deux tells : « tous les niveaux donnent le
   même chiffre » et l'échec du contrôle niveau-0 = drop-modality.
4. **Convention de niveau incohérente par famille** : `is_degraded` traitait 0 dB comme « propre »
   alors que c'est le pire point ⇒ la pire ligne du balayage bruit était en fait le niveau non
   perturbé, présenté comme référence. Corrigé par famille, avec test.
### Addendum — marge de seed de la revendication principale (le caveat de la session 11 est levé)

La session 11 a montré qu'un effet de +0.015 (portail) ne pouvait pas être distingué d'une dispersion
de seed de 0.096. La même question se posait donc pour la revendication **principale** du projet
(« la fusion dépasse le visuel seul »), qui ne reposait que sur **un seed de chaque côté**. Deux seeds
supplémentaires (43, 44) ont été entraînés pour le visuel seul **et** pour `cross`, sur la même config
(30 epochs, grille officielle), et appariés par seed (mêmes conditions machine) :

| seed | `cross` (fusion) | visuel seul | marge |
|---|---|---|---|
| 42 | 0.7745 | 0.7259 | **+0.0486** |
| 43 | 0.7719 | 0.7323 | **+0.0396** |
| 44 | 0.7695 | 0.7278 | **+0.0417** |
| moyenne | 0.7720 | 0.7287 | **+0.0433** (0.0396-0.0486) |

- **Dispersion de seed** : `cross` 0.0050 (0.7695-0.7745), visuel seul 0.0065 (0.7259-0.7323).
  La marge (+0.0433) est donc **≈ 6.7× la dispersion la plus large** ⇒ **la revendication tient** :
  les trois seeds donnent le même signe et restent à ±0.005 de la moyenne. Contraste net avec le
  portail (+0.015 face à 0.096).
- **AP per-clip** : `cross` moyenne 0.7924, plage 0.7847-0.7964. Le « meilleur AP per-clip du projet »
  (0.7961) est donc **le haut de la plage**, pas la valeur centrale — à citer comme 0.792 ± 0.006
  plutôt que comme un point.
- **Enseignement inattendu** : `cross` et le visuel seul sont **très stables** entre seeds (0.005-0.007)
  alors que `adaptive` variait de **0.096**. La famille de modèles n'est donc pas bruyante en soi :
  c'est la **variante à portail** qui l'est. C'est un argument plus solide contre elle que son simple
  classement en AP.
- Conséquence documentaire : la réserve « un seul seed » disparaît pour la revendication principale ;
  elle reste valable pour les rangs intermédiaires de l'échelle (`late`, `early`), non répliqués.
- Outil : `scripts/seed_margin.py` (dispersion par variante + marge appariée par seed, avec verdict
  explicite « marge > dispersion » ; refuse de conclure à n=1). Tableau : `data/logs/seed_margin_table.txt`.


**Règle à retenir** : trois de ces quatre bugs produisaient des nombres plausibles. Ce qui les a
attrapés : (a) un contrôle de cohérence redondant dont la valeur attendue était connue par un autre
chemin, (b) la méfiance devant un résultat trop bon, (c) un test qui aurait dû exister. Un balayage de
robustesse doit donc contenir **au moins une ligne dont la valeur est déjà connue**.

### Artefacts

`runs/2026-09-30_p3vggish30ep_pwlate{,sqrt}` et `_pwcross` (+ `last/` pour cross) :
`test_metrics*.json`, `class_diagnostic.json`, `per_class_localization.json` ; 9 fichiers
`runs/*/robustness_*.json` + `.png` ; tableaux `data/logs/robustness_table.txt` ; journaux
`data/logs/{pw_run,resume_rest,rob_sweeps,redo_sweeps,noise_redo,pw_sqrt}.log` ; 74 tests, `ruff`
propre.

## 2026-10-01 — Session 12 — Scoring causal (P4) : le prix du streaming est mesuré

**Question.** Tous les chiffres du rapport venaient d'une passe **non-causale** : l'attention
agrège les `T` snippets du clip entier. Un système déployé ne le peut pas. Que coûte l'honnêteté ?

**Méthode.** `--causal-prefix-stride K` (`safewatch/eval/runner.py`) re-diffuse des préfixes
espacés de `K` snippets, ne garde que le score à la fin de chaque préfixe et le maintient par
palier ; le dernier préfixe évalué est toujours le clip complet (assertion), donc le snippet `t`
ne voit jamais le futur. **Un seul chemin de scoring** : mêmes datasets, mêmes hooks `drop`/`perturb`
que l'offline — pas de second scorer qui dérive (la leçon de la session 10).

Deux bugs attrapés au passage :
1. `score_clips` n'avait **pas** `@torch.no_grad()` — l'inférence gardait le graphe. Corrigé.
2. Premier lancement du script : le log montrait `EOF while looking for matching quote` — une copie
   `/tmp` obsolète et des processus concurrents (3 scorers en parallèle) faussaient toute mesure de
   débit. Relancé proprement, séquentiel.

**Coût (mesuré, CPU, 4 threads).** `K=1` exact = **182×** la passe offline ; `K=8` = **23×**. Les
trois références se scorent en **6 min 40 s** au total (`cross` 224 s, `late` 86 s, visuel ~76 s) —
bien moins que l'estimation de départ (46.7× le travail-snippet), le petit modèle amortissant le
surcoût par forward.

**Résultat (K=8, grille officielle, 800 clips).**

| run | mode | AP glob. | AP per-clip | F1@0.5 | délai (s) | IoU |
|---|---|---|---|---|---|---|
| `cross` | offline → causal | 0.7745 → **0.6964** | 0.7961 → 0.6118 | 0.239 → 0.208 | 17.7 → 13.5 | 0.260 → 0.297 |
| `late` | offline → causal | 0.7719 → **0.6951** | 0.7857 → 0.6328 | 0.259 → 0.197 | 31.5 → 31.2 | 0.247 → 0.250 |
| visuel | offline → causal | 0.7259 → **0.6521** | 0.7148 → 0.6112 | 0.201 → 0.181 | 35.3 → 31.6 | 0.208 → 0.240 |

Premières lectures (seed 42 seul) : la localisation semblait *s'améliorer* et la marge de fusion
*tenir*. **Un contrôle de seed a corrigé la première.** Voir l'addendum ci-dessous — c'est la
partie qui compte.

Ce qui souffre, c'est le **classement intra-clip** (AP per-clip −0.10 à −0.19) : maintenir le score
par palier crée des ex-æquo *dans* un clip, et l'AP per-clip mesure exactement ce classement. C'est
la conséquence attendue de `K=8`, pas un défaut du modèle — `K=1` la supprimerait au prix de 182×.

### Addendum — contrôle de seed : une de mes conclusions était fausse

La marge `cross − visuel` offline est répliquée sur 3 seeds ; la conclusion causale ne pouvait donc
pas rester à n=1. J'ai scoré en causal les quatre runs de seed (`cross`/`visual` en `s43`/`s44`) via
`TAGS="..." bash scripts/run_causal_score.sh` (~7 min) et étendu `scripts/seed_margin.py` d'un
`--metrics-suffix` pour lire `test_metrics_causal8.json`. Verdict, cellule par cellule
(`data/logs/causal_seed_matrix.txt`) :

| effet du streaming | plage sur 6 cellules (3 seeds × 2 modèles) | robuste ? |
|---|---|---|
| ΔAP globale | −0.0717 … −0.0901 | **oui** (même signe partout) |
| ΔAP per-clip | −0.1025 … −0.1906 | **oui** |
| ΔF1@0.5 | −0.0198 … −0.0494 | **oui** |
| ΔIoU | **−0.0973 … +0.0368** | **non — signe inversé** |
| Δdélai | **−4.2 s … +7.9 s** | **non — signe inversé** |

1. **Le coût en AP est robuste** — ~10 % d'AP en moins, systématiquement, sur les 6 cellules.
2. **Ma lecture « la localisation ne souffre pas » était un artefact de la seed 42.** Sur les seeds
   43/44 l'IoU *baisse* (−0.08 à −0.10) et le délai *monte* (+5 à +8 s) — exactement l'inverse.
   Et le délai offline lui-même vaut **17.7 / 3.5 / 0.2 s** selon la seed (dispersion 17.5 s) :
   c'est une métrique **instable**, à ne pas citer à n=1. Leçon : la leçon de la session 11
   (marge vs dispersion) s'applique aussi aux métriques de *localisation*, pas seulement à l'AP.
3. **La marge de fusion tient, mais s'affaiblit** : +0.0334 en causal (positif sur les 3 seeds,
   range +0.0245…+0.0442) contre +0.0433 en offline. Le rapport marge/dispersion passe de **6.7×
   à 2.3×**, et la dispersion causale de `cross` (0.0145) est **3×** l'offline (0.0050) : **le
   streaming rend le modèle plus sensible à la seed**. La valeur que j'avais citée au premier jet
   (+0.0442) était la plus optimiste des trois.

**Conclusion corrigée** : le modèle est déployable en flux à ~10 % d'AP près (chiffre robuste) ;
l'effet du streaming sur la localisation n'est **pas** établi et ne doit pas être présenté comme un
gain. Rapport §4.5 et §6.4 mis à jour en conséquence.

### Artefacts
`scripts/run_causal_score.sh` (`TAGS=` pour cibler d'autres runs) ; `scripts/seed_margin.py`
(`--metrics-suffix`) ; `runs/*/predictions_causal8.npz` + `test_metrics_causal8.json` pour
`p3vggish30ep_cross`, `p3vggish30ep_late`, `p2vggish30ep_visual` et les seeds `_s43`/`_s44` ; tables
`data/logs/causal_table.txt` + `data/logs/causal_seed_matrix.txt` ; journaux
`data/logs/causal{,_seeds}.log` ; 4 tests ajoutés (89 au total), `ruff` propre.

## 2026-10-01 — Session 13 — L'échelle de fusion : `adaptive` ne battait pas le visuel seul

**Le déclencheur.** En relisant le tableau §2 du rapport avec la leçon de la session 12 en tête
(« une marge doit dépasser la dispersion »), une ligne a sauté : `adaptive` y figure à **0.7281**,
juste au-dessus du visuel seul (0.7259). Or la session 11 avait mesuré que l'`adaptive` oscille de
**0.096** entre seeds. 0.7281, c'était donc *sa meilleure seed*, présentée comme son résultat.

**Ce que les fichiers disaient.** Les seeds 43/44 de l'`adaptive` existent bel et bien — mais elles
n'ont jamais eu de `test_metrics.json` : la réplication du portail les a seulement passées au
**balayage d'occlusion**, qui écrit la valeur *propre* comme **level 0** du fichier de sweep. D'où
le script `scripts/ladder_seed_check.py`, qui va chercher l'AP propre là où il est réellement
(`test_metrics.json`, sinon `level == 0.0` d'un `robustness_*_blind.json`) — `ap_global` étant un
PR-AUC au niveau snippet, la règle segmentaire n'y entre pas, donc les deux sources sont comparables.

**Résultat (§2, `data/logs/ladder_seed_table.txt`) :**

| variant | n | moy. | plage | dispersion | vs visuel seul |
|---|---|---|---|---|---|
| `cross` | 3 | 0.7720 | 0.7695–0.7745 | 0.0050 | **+0.0433** |
| `late` | 1 | 0.7719 | — | — | +0.0432 |
| `early` | 1 | 0.7643 | — | — | +0.0356 |
| `adaptive` | 3 | **0.6900** | **0.6319–0.7281** | **0.0962** | **−0.0387** |
| visuel seul | 3 | 0.7287 | 0.7259–0.7323 | 0.0065 | — |
| audio seul | 1 | 0.6403 | — | — | −0.0884 |

Donc la phrase « toutes les fusions battent la référence visuelle (+0.04 à +0.05) » était **fausse
sur une ligne** : à 3 seeds l'`adaptive` est **0.039 _sous_** le visuel seul, et sa marge per-clip
(+0.015) tient dans sa propre dispersion (0.057). La plage annoncée (« +0.04 à +0.05 ») excluait
d'ailleurs cette ligne — le quantificateur et la plage se contredisaient.

**Corrections.** §2 : colonne « seeds » ajoutée (n + plage + moyenne), ligne `adaptive` annotée ⚠️,
phrase réécrite pour nommer `cross`/`late`/`early` au lieu de « toutes ». §2.2 : « la fusion bat sa
référence » ne vaut plus que pour ces trois-là. README idem. Script + table ajoutés au dépôt, plus
un test (`clean_ap` doit trouver `level == 0.0`, pas `levels[0]` — les sweeps réels sont ordonnés,
mais s'y fier serait une bombe à retardement).

**Ce que ça change pour le fond.** Rien sur la revendication centrale (`cross` bat le visuel seul de
+0.043, 3 seeds, 6.7× la dispersion) ; mais ça retire le portail du tableau des « gains » **par deux
chemins indépendants** : il ne gagne pas (cette session) et il n'est pas stable (session 11). Deux
erreurs de la même famille en trois sessions — une métrique citée à n=1 — donc la règle passe au
premier plan : *avant de citer un chiffre, dire combien de seeds le portent*.

### Artefacts
`scripts/ladder_seed_check.py` ; `data/logs/ladder_seed_table.txt` ; §2 et §2.2 du rapport ;
README P3 ; 1 test ajouté (90 au total), `ruff` propre.

## 2026-10-01 — Session 14 — P5 : la calibration de la tête catégorielle, trois méthodes, zéro gain

**Question laissée ouverte en session 11.** La température **globale** améliorait la NLL sans
toucher l'ECE. Restait à tester la calibration **par classe** — la carte d'alerte affiche une
*catégorie* (« riot 91 % »), donc c'est la tête multi-label qu'il faut calibrer, pas le score
d'anomalie clip.

**Implémentation.** `calibration.PerClassCalibration` + `fit_per_class` (une température par
catégorie, ajustée sur les faibles étiquettes clip de la val) et `fit_calibration.py --per-class`,
qui collecte les deux têtes en une passe val et réutilise `runner.score_clips` pour le test (un
seul chemin de scoring). Détail qui compte : une catégorie **absente** de la split d'ajustement ne
peut pas être calibrée (`fit_temperature` refuse une cible mono-classe) — elle est donc listée dans
`unfitted` avec T=1, jamais silencieusement ignorée.

**Résultat (ECE mesuré sur 800 clips test, T ajusté sur 150 val ; `data/logs/perclass_calib_table.txt`) :**

| macro ECE test | non calibré | par classe | T globale unique |
|---|---|---|---|
| `cross` | **0.0738** | 0.0810 | 0.1100 |
| `late` | 0.0426 | **0.0425** | 0.0676 |

1. **La température globale est franchement nuisible** à cette tête (+0.036 sur `cross`, +0.025 sur
   `late`) : elle a été ajustée sur le score d'anomalie *clip* et sur-écrase les 6 probabilités de
   catégorie. C'est le genre d'erreur qu'on n'aurait pas vue en ne regardant que la NLL.
2. **La calibration par classe ne rattrape pas** : sur `cross` elle *dégrade* l'ECE (4 têtes
   aggravées sur 5 ajustées), sur `late` c'est une égalité à 1e-4. Aucune des deux options n'égale
   « ne rien faire ».
3. **`abuse` est incalibrable** : 0/150 positifs en val. T=1 imposé, `unfitted` remonté — condition
   pour qu'un opérateur ne prenne pas « abuse 12 % » pour une probabilité.

**Cause racine, écrite noir sur blanc** : la mise à l'échelle par température minimise la **NLL**,
pas l'**ECE** ; l'ECE n'est qu'un sous-produit et peut donc empirer *même sur la split
d'ajustement* (`cross` val : macro 0.0755 → 0.0774). Viser l'ECE demande une méthode dont c'est
l'objectif (isotone / histogram binning) — noté, hors périmètre.

**Décision utilisable** : on **laisse la tête catégorielle non calibrée** et la carte présente la
catégorie comme un **rang**, pas comme une probabilité. C'est plus honnête qu'un T appris sur 150
clips qui n'améliore rien.

### Artefacts
`safewatch/eval/calibration.py` (`PerClassCalibration`, `fit_per_class`) ;
`scripts/fit_calibration.py --per-class` ; `runs/*/calibration_per_class.json` (cross, late) ;
table `data/logs/perclass_calib_table.txt` ; journal `data/logs/perclass_calib.log` ; 3 tests
ajoutés (93 au total), `ruff` propre.

## 2026-10-04 — Session 15 — AED (YAMNet gelé, ONNX) : l'alerte nomme les événements sonores

**La lacune.** Le sujet exige la « reconnaissance d'événements acoustiques » et que chaque alerte
présente les « événements sonores identifiés ». L'audio était décrit jusqu'ici par des
*statistiques de signal* (rms, flatness, zcr, saturation) : la carte pouvait dire « bruit large
bande, sature », jamais « explosion », « tir » ou « crissement ». La docstring de `report.py`
disait d'ailleurs noir sur blanc qu'aucun AED n'était installé.

**Choix du modèle — et pourquoi.** Trois voies étaient ouvertes :
1. *PANNs CNN14* (41 M paramètres, mAP AudioSet 0.431) : ~0,5–1,5 s par clip de 10 s sur le CPU,
   soit des heures sur la split test. Rejeté : pas compatible avec le temps quasi réel.
2. *YAMNet en TF* (4,2 M paramètres) : bon temps CPU, mais TF sur Python 3.14 n'existe pas et
   ferait 1 Go de dépendance pour un pas de prétraitement figé.
3. **YAMNet en ONNX** (16,1 Mo, `jafet21/yamnetonnx` sur HF, poids officiels Google convertis) :
   `onnxruntime` est déjà dans le venv. **Retenu** : aucun nouveau package, ~50 ms par 6,7 s
   d'audio mesurés sur la machine du projet (~8 ms par image de 0,96 s) — temps réel avec large
   marge.

**Validation du modèle avant de l'emmurer.** (i) *Smoke test* : `miaow_16k.wav` officiel →
`Animal 0.71, Cat 0.42, Meow 0.20, Caterwaul 0.09` — le bon chat. (ii) *Alignement des images,
mesuré, pas supposé* : la sortie ONNX émet `max(1, n_samples // 8000)` images ; une rafale de
440 Hz à 0,2–0,4 s dans 2 s d'audio n'allume que l'image 0, et une rafale à 0,7–0,9 s allume les
images 0 **et** 1 ⇒ l'image *i* couvre `[0,5i, 0,5i+0,96]` s (fenêtre 0,96 s, pas 0,5 s, queue à
zéro). C'est cette règle qui figure dans `safewatch/edge/aed.frame_overlapping`, et le test de
rafale est figé dans `tests/test_aed.py`.

**Découverte qui a failli coûter la session.** Le `vggish_model.ckpt` officiel (291 Mo, GCS) est
**uniquement l'embedding** : le README officiel dit qu'on a remplacé la tête 1000-classes par une
tête 128-d. **Il n'existe pas de tête tagger VGGish publique** — l'idée d'un « tagger VGGish à
coût nul sur nos features » est morte là. (Le `torchvggish` du venv et le ckpt officiel contiennent
exactement les mêmes 72 M paramètres, vérifié.) D'où YAMNet.

**Pipeline.** `safewatch/edge/aed.py` (module figé : 521 classes, ensemble curé de 16 classes
« signature d'incident » documenté par catégorie) + `scripts/extract_aed_tags.py` (YAMNet sur
l'audio brut de chaque clip, **aucun** ré-entraînement, ~1,7 s/clip ffmpeg compris, reprenable) →
`data/features/aed/tags/<clip>.npz` (T×521 float16, ~50 Mo pour la split test). La carte
(`explain/report.py`) lit ces tags : par segment, les classes curées sont rangées par leur
**pic** dans le segment (pas la moyenne qui noyerait un explosion de 2 s dans 45 s de dialogue),
avec l'heure du pic ; l'« incident_energy » (moyenne sur les 16 classes) est la fraction du
segment qui *sonne comme un incident*. Deux honnêtetés codées : le caveat bascule avec les tags
(tagger absent → « signal statistics only »), et l'événement nommé est présenté comme
**corroboration** (`corroboration_only: true`) — le tagger n'a jamais vu XD-Violence.

**Résultats mesurés.**
* Clip `Bad.Boys.1995` (explosion|shooting|car_accident) : `Explosion (0.73 @ 00:12), Engine
  (0.23 @ 00:26), Smash, crash (0.23 @ 00:32)` — et les heures tombent **dans les intervalles
  annotés** (12,7 s ∈ 10,4–15 s ; 26,7–32 s ∈ 24,3–33,8 s).
* Clips normaux : **zéro** événement nommé, et la carte le dit à voix haute
  (« aucun événement sonore d'incident nommé ») — pas de boîte sur du silence.
* **AP de la signature d'incident** (16 classes curées, moyenne par snippet) contre la vérité
  terrain par snippet, 500 clips test annotés : **0,6787 global / 0,6550 per-clip**
  (`data/features/aed/aed_stats.json`). C'est un signal de corroboration correct, pas un détecteur
  d'incidents : on ne l'affiche jamais comme le motif de l'alerte.

**Coupé au scalpel (leçon de la session).** Le premier `frame_overlapping` avait un off-by-one :
`floor((start − 0,96)/0,5)` au lieu de `floor(...) + 1` (la borne basse est une inégalité
**stricte**) — les tags auraient glissé d'un demi-frame sur chaque snippet, avec des nombres
plausibles à la clé. Attrapé par le test `test_frame_overlapping_uses_the_measured_window`,
avant tout résultat.

### Artefacts
`safewatch/edge/aed.py` (+ export dans `safewatch/edge/__init__.py`) ; `scripts/extract_aed_tags.py` ;
`scripts/check_vggish_space.py` (audit d'espace des features VGGish : brutes pré-PCA, corrélation
0,84–0,88 vs 0,29–0,39 post-PCA) ; `data/features/aed/{yamnet.onnx, yamnet_class_map.csv,
tags/ (800 npz), aed_stats.json}` ; `explain/report.py` (`audio_events`, `acoustic_events` dans la
carte, caveats) ; `scripts/incident_sheet.py` (chargement des tags) ; `ui/state.py` (YAMNet
on-the-fly sur les uploads) ; `app.py` (affichage des zones) ; 16 tests AED + 4 tests
salience, `ruff` propre.

---

## 2026-10-05 — Session 16 — Campagne de remède sur les sept négatifs honnêtes

Objectif : pour chaque négatif documenté, un résultat mesuré nouveau ou une preuve documentée
d'irréparabilité. Deux expériences lourtes tournent en fond ce jour : CLIP K=2 mean+std
(`scripts/run_clip_head.sh`, `data/logs/clip_k2.log`) et la matrice de remède
(`scripts/run_remedy_matrix.sh` : gate de fiabilité 3 seeds + cross sur-échantillonné).

**#2 Calibration — l'isotonique casse le cul-de-sac de la tête catégorielle.** Session 14 avait
établi que la température (globale, puis par classe) ne réduisait pas l'ECE de la tête multi-label.
Rung non essayé : la **régression isotonique** (monotone non-paramétrique, `sklearn`).
Mesuré sur `p3vggish30ep_cross_s43` (grille officielle, 150 val / 800 test) :

| tête | méthode | ECE macro test (honnête) |
|---|---|---|
| catégorie (6 têtes) | température par classe | 0.0732 -> **0.0892** (pire) |
| catégorie (6 têtes) | **isotonique par classe** | 0.0732 -> **0.0373** (divisé par deux) |
| binaire clip | isotonique | 0.0268 -> 0.0533 (pire à la test) |

Par classe à la test, isotonique : fighting 0.0881→0.0516, shooting 0.0932→0.0385, riot 0.0455→0.0246,
car_accident 0.0616→0.0168, explosion 0.0775→0.0547 ; abuse reste **non calibrable** (0 positif en
val — le nœud #7, pas un défaut de calibrage). La tête binaire, elle, est déjà quasi calibrée
(ECE 0.027) : l'isotonique y sur-adjuste et perd à la test. **Décision : isotonique par classe pour
la confiance affichée par catégorie (la carte), température pour la tête clip.** Code :
`calibration.fit_isotonic / apply_isotonic / fit_per_class_isotonic`, `fit_calibration.py
--method isotonic`. Piège de version évité : `IsotonicRegression` (sklearn 1.9) expose les
points de cassure via `f_.x / f_.y`, pas `X_`/`y_` — le mapping stocké est vérifié au test comme
identique au `predict` clipé.

**#3 int8 — rejet n°4, mais la voie de reduction sans drift est prouvée : fp16.** La quantification
dynamique ORT sans calibration reste trop grossière pour ce graphe : **per-channel** (une échelle
par canal de sortie, contre per-tensor au rejets 1-3) réduit le dérive des logits de
**0.13 -> 0.0799**, toujours > 0.05 => rejeté, et c'est mesuré, pas supposé (1.41 Mo, 0.345× la
taille, 1.17× plus rapide en p50). Ce qui manquait aux 3 rejets : le **float16**. Export fp16
construit dans `export_onnx.py --fp16` (poids en fp16 + Cast aux bornes du graphe ; le graphe
tracé de MHA portait des `Cast→FLOAT` et une `ConstantOfShape` -inf restés en fp32, convertis —
sans ça ORT refuse le graphe) : **dérive 0.0038 => accepté**, taille 0.508×, p50 5.54 ms contre
6.80 ms fp32 (1.23×, CPU 4 threads, charge ~11 avec le job CLIP concurrent). Résumé : int8 rejeté
4 fois sur 4, fp16 est la variante de réduction de taille qui garde la parité — et c'est maintenant
un artefact mesuré (`export/onnx/..._fp16.onnx` + report), pas un argument. Bug d'export corrigé au
pass : `strip_attention_weights` ignorait le `reliability` (modulation SNR de la branche audio
silencieusement absente du graphe).

**#5 Normalisation audio par clip — preuve documentée d'irréparabilité, le vrai verrou était ailleurs.**
Mesuré en Session 9e : audio seul 0.446→0.329 (per-clip 0.625→0.593), late 0.7825→0.7703 per-clip.
La normalisation d'instance **détruit le niveau sonore absolu, qui est discriminant** (une explosion
est *forte* ; un dialogue, silencieux). Sa motivation d'origine — les scores audio non comparables
**entre** clips — est depuis traitée par la bonne voie : features VGGish 128-d sur le corpus
complet (Session 10) + z-score par dimension avec **stats globales de la split train**
(`scripts/audio_stats.py` → `data/features/vggish_stats.json`, `audio_norm="none"`) + calibration
au niveau décision (P5). L'option `--audio-norm clip` reste dispo (documentée, testée, mesurée
négative) ; le pipeline expédie la combinaison qui a gagné. Néant ferme, avec les chiffres.

**#1, #4, #6, #7 — en cours, chiffres à l'arrivée des jobs :**
* #1 : `--gate-source reliability` (la porte ne voit que les 7 canaux de fiabilité, plus
  d'embeddings => pas de co-adaptation aux embeddings d'un seed) — 3 seeds en file.
* #4 : CLIP K=2 mean+std (1024-d) en extraction ; le test K=1 montrait le gouffre sur les classes
  de mouvement (fighting -0.18).
* #7 : `--class-over-sample auto` (WeightedRandomSampler, exposition des 50 clips abuse),
  cross s43.
* #6 : seeds croisés s45/s46 ensuite (localisation causale).


**Résultats mesurés — reliability-gate adaptive, 3 seeds (grille officielle, 800 test clips) :**

| seed | global PR-AUC | per-clip PR-AUC | macro per-class AP |
|---|---|---|---|
| s42 | 0.7589 | 0.7700 | 0.6703 |
| s43 | 0.7422 | 0.7361 | 0.6061 |
| s44 | 0.7307 | 0.7485 | 0.4042 |
| **moyenne** | **0.744** | **0.751** | — |
| ancien embedding-gate (3 seeds) | 0.6900 | — | — |

Spread 0.028 contre 0.096 pour l'embedding-gate ; les 3 seeds sont au-dessus du visuel seul (0.7259–0.7323), où l'ancien s43 descendait à 0.094 en dessous. L'hypothèse de co-adaptation aux embeddings est confirmée par les chiffres : le verrou #1 est résolu.

**État des jobs au 2026-10-06T08:00 :**
* #7 cross_osample_s43 : reprise nécessaire (ckpt_best à epoch ≤16, pas de ckpt_last).
* #4 CLIP K-2 : 3383/3804 features extraites en cache — extraction reprise.
* #6 causal seeds s45/s46 : pas encore lancé.

**Résultats mesurés — reliability-gate adaptive, 3 seeds (grille officielle, 800 test clips) :**

| seed | global PR-AUC | per-clip PR-AUC | macro per-class AP |
|---|---|---|---|
| s42 | 0.7589 | 0.7700 | 0.6703 |
| s43 | 0.7422 | 0.7361 | 0.6061 |
| s44 | 0.7307 | 0.7485 | 0.4042 |
| **moyenne** | **0.744** | **0.751** | — |
| ancien embedding-gate (3 seeds) | 0.6900 | — | — |

Spread 0.028 contre 0.096 pour l'embedding-gate ; les 3 seeds sont au-dessus du visuel seul (0.7259–0.7323), où l'ancien s43 descendait à 0.094 en dessous. L'hypothèse de co-adaptation aux embeddings est confirmée par les chiffres : le verrou #1 est résolu.

**État des jobs au 2026-10-06T08:00 :**
* #7 cross_osample_s43 : reprise nécessaire (ckpt_best à epoch ≤16, pas de ckpt_last).
* #4 CLIP K-2 : 3383/3804 features extraites en cache — extraction reprise.
* #6 causal seeds s45/s46 : pas encore lancé.
