# SafeWatch Edge AI — détection multimodale, robuste et explicable d'incidents dans les flux audio-vidéo

Weakly-supervised **audio-visual** incident detection on surveillance-style streams: detect, categorise,
temporally localise, explain, and run on an Edge budget. Benchmark dataset: **XD-Violence** (ECCV 2020).

## Status

| Phase | Scope | State |
|---|---|---|
| P0 | Environment + repo skeleton | ✅ done |
| P0 | Data bootstrap + audio check + stride calibration | ✅ done |
| P1 | Evaluation harness + localisation metrics (`safewatch/eval/metrics.py`) | ✅ done · 107 tests green (14 metrics + 41 pipeline + 20 explain + 10 edge + 8 analysis + 14 UI) |
| P1 | Feature grids verified: mirror 64 frames, **official 16 frames** · dataset, losses, models, train/score CLIs | ✅ done |
| P1 | Visual-only MIL baseline, mirror features (800 test clips) | ✅ AP global **0.813** / per-clip 0.764 |
| P1 | Visual-only MIL baseline, **official I3D features** (16-frame grid) | ✅ AP global **0.722** / per-clip 0.706 — weaker than the mirror's VideoSwin features, and under-tuned for the finer grid (`docs/journal.md`, Session 8) |
| P4 | Per-category localisation (`scripts/per_class_localization.py`) | ✅ incidents are located per category, at a **wall-clock** segment rule. Category-adaptive thresholds and durations (`car_accident` 1.5s, `riot` 4s with lookback) integrated into metrics & UI. |
| P1 | Source-disjoint audit | ✅ AP is **not** inflated by source overlap: unseen-source clips score **better** on every run (mirror late 0.864 vs 0.745 swin; official 0.787 vs 0.672; **30ep official-grid fusion `cross` 0.8452 vs 0.7435 global, 0.8258 vs 0.7817 per-clip** — 66.6 % of test clips share a source with train) — `test_metrics_source_disjoint.json` per run |
| P2 | Audio features + audio-only baseline | ✅ mel 132-d **4 750 / 4 750** clips (the last missing test mp4, `Before.Sunset.2004__#01-14-30_01-16-51_label_A`, was recovered via direct LFS download — the Hub metadata size is stale — and mel extracted; `audio_stats.json` now full-train 3 516 clips / 231 509 snippets). VGGish 128-d: **official release bridged 4 754 / 4 754** (`vggish_snippet/` symlinks, 0 off-grid) + `vggish_stats.json` (official train 3 804 clips). **Official-grid baselines (correct snippet alignment): audio-only VGGish 0.619 global / 0.691 per-clip, visual-only I3D 0.671 / 0.703** |
| P5 | Weak-class diagnostic (`scripts/class_diagnostic.py`) | ✅ **`abuse` is starved, not broken**: only **50 / 3 804** train clips (1.3 %; every other class has 367–463). Calibrated per-class operating threshold ($\theta_\text{abuse} = 0.10$) and percentile ranking integrated into reporting. |
| P5 | Class re-weighting (`--multi-pos-weight auto\|sqrt`) | ❌ **the diagnosis was right, the remedy is not**: inverse-frequency weighting (abuse weight 75) **revives the dead head** (0 → 7/11 TP, p(abuse) 0.069 → 0.633) but for a *ranking* metric it trades ranking for recall — `late` abuse AP **0.138 → 0.096**, all heads over-fire (FP ×2–×4), and damping (`sqrt`, weight 8.7) does not fix it (0.094). On `cross` it helps a lot (+0.169 abuse AP, macro +0.089, for −0.008 global) ⇒ the effect **interacts with the fusion mode**, so one run would have lied. Side effect on both: mean IoU +0.11 (single seed, confounded). Also found the official **val split contains no `abuse` clip at all** ⇒ model selection was structurally blind; `ckpt_last.pt` is now saved and is the better categorisation checkpoint (abuse 0.259 vs 0.250, macro 0.790 vs 0.785) |
| P3 | Ablation ladder on the **full corpus**, **mirror grid** (VideoSwin 768-d + log-mel 132-d, 30 epochs, same list) | ✅ **late fusion wins**: 0.8125 global / 0.7825 per-clip vs visual-only 0.8100 / 0.7709. `cross` 0.7981, `adaptive` 0.7553, `adaptive+clipnorm` 0.7668 ⇒ gating novelty does **not** pay on these features |
| P3 | Ablation ladder on the **official grid** (I3D 1024-d + real VGGish 128-d, 30 epochs, same 800 test clips) | ✅ **fusion wins by a wide margin for `cross`/`late`/`early`**: `cross` **0.7745 global / 0.7961 per-clip** vs visual-only 0.7259 / 0.7148 (**+0.0486 / +0.0813**, ≈20× the mirror margin) · `late` 0.7719 / 0.7857 · `early` 0.7643 / 0.7830 · audio-only 0.6403 / 0.6986. ⚠️ **`adaptive` is NOT a win**: its 0.7281 is its **best of 3 seeds**; the 3-seed mean is **0.6900**, *below* the visual-only baseline (0.7287), with a 0.096 spread (`scripts/ladder_seed_check.py` → `data/logs/ladder_seed_table.txt`). Removing audio costs **−0.050** global, removing video **−0.118** ⇒ audio genuinely contributes (no longer a "passenger"). Best per-clip AP of the project: **0.7961**. Absolute global AP remains below the mirror grid (0.8125) because official I3D is the weaker visual backbone (0.7259 vs 0.8100 visual-only). Fair-wall-clock localisation (5.33 s min segment, matching the mirror rule): `cross` IoU **0.369** / F1@0.5 0.243 / delay **5.4 s** vs mirror 0.423 / 0.306 / 11.4 s |
| P3 | **Seed margin of that claim** (`scripts/seed_margin.py`) | ✅ **the margin holds**: 2 extra seeds trained for both `cross` and the visual-only baseline, paired by seed, give **+0.0486 / +0.0396 / +0.0417** (mean **+0.0433**). Seed spread is **0.0050** (`cross`) and **0.0065** (visual), so the margin is **~6.7× the wider spread** ⇒ not seed noise (unlike the gate: +0.015 against a 0.096 spread). Per-clip AP should be quoted as **0.792 ± 0.006** (range 0.7847–0.7964), not as the single best 0.7961. Bonus: `cross`/visual are very seed-stable while `adaptive` swings 0.096 ⇒ the gated variant is the unstable one, not the model family |
| P3 | **Score-level ensemble** of the non-gated fusion variants + seeds (`scripts/ensemble_scores.py`) | ✅ **the project's best configuration — accuracy for free** (no new backbone, no GPU, no extra features; members share the input features, so the edge cost is only N tiny heads ≈ 6.5 ms each). Two levels: (i) the three non-gated **fusion variants** of one seed (`late`+`cross`+`early`, s42) → **global AP 0.8125 → 0.8216**, bootstrap 95 % CI [+0.0053, +0.0138], P(Δ>0)=1.000; (ii) adding **seed diversity** (all three `late` seeds, **no cherry-picking**) → `late(42,43,44)`+`cross42`+`early42` = **global AP 0.8241** (**+0.0116** vs `late42`; bootstrap **+0.0113**, 95 % CI **[+0.0076, +0.0155]**, P(Δ>0)=**1.000**), per-clip **0.7642** and **F1@IoU.5 0.3193 vs 0.3064** — note the *3-member* version dipped to 0.2946, so seed diversity also removes the fixed-threshold F1 regression. Context: the `late` seed spread alone is **0.8125–0.8206** (±0.008), so single-seed comparisons carry that uncertainty and the ensemble gain exceeds it. `adaptive` is deliberately excluded (weakest, least seed-stable) · runs `runs/2026-10-04_p3full_ens_final` (5-member) and `..._ens_lce` (3-member) |
| P3 | **Visual-representation swap** (`scripts/extract_clip.py`, `scripts/make_concat_features.py`) | ⚠️ **two honest negatives, both diagnosed**: a frozen CLIP ViT-B/32 visual stream on the mirror grid scores **0.7688** global vs VideoSwin **0.7981** (controlled — same lists, audio, head and hyper-parameters; only the visual stream changes 768→512-d). The loss lands on the **motion** classes (`fighting` **−0.181**, `explosion` **−0.096**) while the semantic ones are level or better (`shooting` +0.031, `abuse` +0.019), because CLIP sees **one static frame per 2.67 s snippet** whereas VideoSwin summarises a 64-frame window. Concatenating the two views (768+512 = 1280-d, `--visual-dim 1280`) scores **0.7995** — the head does **not** exploit the complementarity (it lifts `abuse` 0.017→**0.068** and `riot`, but dilutes `fighting` 0.841→0.791). As an ensemble member it merely trades global for per-clip (0.8186 / **0.7691** vs 0.8216 / 0.7652) ⇒ **the ensemble of same-feature fusion variants remains the best configuration**. Fix for the motion gap is implemented and unrun: `--frames-per-snippet K --pool mean+std` (step 1b) |
| P3 | **Audio-representation swap: VGGish pooled onto the mirror grid** (`scripts/pool_vggish_mirror.py`) | ⚠️ **neutral, not a win**: the official grid's audio win came from real VGGish 128-d (audio went from passenger to load-bearing, +0.0486), so the mirror grid — which kept the weaker log-mel 132-d — was the obvious place to retry it. 4 official snippets fold into 1 mirror snippet (16 vs 64 frames) by windowed mean, giving `data/features/vggish_mirror` (4 750 clips, 5 s). Same head, same lists: **0.8103 global / 0.7389 per-clip** vs log-mel **0.8125 / 0.7621** ⇒ global is a wash (−0.0022) and per-clip is **worse** (−0.0232), though F1@IoU.5 improves (0.3211 vs 0.3064). Per-class it is a trade: `abuse` +0.034, `riot` +0.014, `fighting` +0.003 vs `shooting` **−0.092**, `explosion` −0.032. As an ensemble member it adds +0.0017 global (0.8233) but costs per-clip (0.7594) — **within noise**, so the 3-member ensemble stands |
| P3 | Audio cross-clip calibration hypothesis | ❌ **refuted** (honest negative result): per-clip instance norm costs per-clip AP (0.7825 → 0.7703) and destroys audio-only (0.446 → 0.329); removing audio still does not hurt the mirror-grid global score. But the follow-up **confirmed the real bottleneck**: swapping log-mel 132-d → official VGGish 128-d (see row above) grew the audio contribution from +0.0025 to **+0.0486** global and made both modalities necessary ⇒ the plateau was the **audio representation**, not the fusion architecture |
| P4 | Temporal localisation (segments in **seconds**) + online mode | ✅ segment rule + tIoU/delay metrics; **wall-clock rule used for cross-grid comparisons** (`--min-seconds`/`--max-gap-seconds`; the snippet-based default silently gave a 4× finer rule on the 16-frame grid). **Causal/streaming scoring done** (`--causal-prefix-stride K`, `scripts/run_causal_score.sh`): strided prefix re-forwards keep only each prefix-end score so snippet *t* never sees the future. **Seed-controlled (3 seeds × 2 models, `data/logs/causal_seed_matrix.txt`)**: the AP cost is **robust** — global AP **−0.072 to −0.090**, per-clip **−0.10 to −0.19**, F1@0.5 **−0.020 to −0.049**, same sign in all 6 cells. The **localisation effect is NOT robust** — IoU (**−0.097 to +0.037**) and delay (**−4.2 s to +7.9 s**) **flip sign** across seeds, and the offline delay itself swings **0.2–17.7 s** ⇒ no localisation claim is citable at n=1. The fusion-over-visual margin survives but weakens: **+0.0334** causal (3 seeds) vs **+0.0433** offline; margin:spread drops **6.7× → 2.3×**. Cost: K=8 = 23× offline (K=1 exact = 182×); 7 runs in ~13 min CPU |
| P8 | Robustness study (degraded / missing modality) | ✅ **four degradation families swept** (`scripts/robustness_sweep.py`, `scripts/robustness_table.py`; **feature-space** perturbations — see caveat below). Audio → silence costs `cross` **−0.050 APglob / −0.100 macro** but **−0.001** for the mel model (it never used audio) ⇒ the VGGish representation is what makes audio *load-bearing*. Additive noise at **0 dB (noise = signal power) is harmless** (+0.002 APglob, −0.011 APclip) while losing 75 % of audio snippets costs −0.015 ⇒ **information deletion, not perturbation, is the failure mode**. Visual occlusion 30 s: `cross` −0.012, `late` −0.024, `adaptive` −0.009. **The reliability gate: the replication does *not* confirm it.** Told the visual stream is degraded, `adaptive` gained **+0.016 / +0.020 AP** over its own blind run on the original seed (15 s / 30 s). Two extra seeds, same config, different occlusion placement, give **+0.016 / −0.004 / +0.031** (15 s) and **+0.020 / −0.002 / +0.037** (30 s) ⇒ **2 of 3 positive**, mean +0.015/+0.018, one strictly null. Worse: across those three seeds the **clean** `adaptive` model itself spans **0.6319–0.7281 (spread 0.096, σ 0.051)** — six times the claimed effect, which therefore cannot be separated from seed noise. It is also the most seed-unstable of the four variants. Verdict: the gate is **not refuted but not demonstrated**; it must not be presented as a gain |
| P8 | Robustness caveat (what these numbers are) | ⚠️ **feature-space, not media**: the release ships precomputed features; only log-mel can be regenerated from the videos (~3 s/clip measured ⇒ ≈2.5 h for a 4-level SNR sweep on 800 clips). VGGish/I3D extractors are not available here, so these are *representation* robustness results. Three of the four bugs found this session produced plausible numbers; what caught them was a redundant consistency check with a known expected value |
| P5 | Explanations, calibration, incident sheets | ✅ sheets done; calibration is a **measured dead end on the category head**: the single global temperature *hurts* it (`cross` macro ECE test **0.0738 → 0.1100**, `late` **0.0426 → 0.0676** — it was fitted on the clip-level anomaly score), and **per-class** temperatures (`--per-class`, 6 heads) do not rescue it: `cross` **0.0810** (worse than nothing, 4 heads of 5 aggravated), `late` **0.0425** (a 1e-4 tie). `abuse` is **uncalibratable**: the official val split has **0/150** `abuse` clips ⇒ no positive ⇒ T=1, flagged `unfitted` rather than hidden. Root cause stated: temperature scaling minimises NLL, **not** ECE, so ECE can worsen even on the fitting split (`cross` val macro 0.0755 → 0.0774). Practical rule adopted: leave the category head uncalibrated and show the category as a *rank* (`data/logs/perclass_calib_table.txt`) |
| P5 | **Acoustic event recognition** (frozen AED, fills the subject's "reconnaissance d'événements acoustiques") | ✅ **YAMNet** (AudioSet-YouTube, 521 classes, 4.2 M params) as a frozen ONNX tagger on the existing audio: named events with measured probability + time on every alert — explosion clips read `Explosion (0.73 @ 00:12), Engine (0.23), Smash, crash (0.23)` and the named events land **inside the annotated GT intervals**; normal clips get **no named event** (the card says so out loud). Runs on the CPU at ~8 ms per 0.96 s frame (~1 s/clip total incl. ffmpeg) — real-time for the edge track. The tagger was **never trained on XD-Violence**, so the card presents it as *corroboration* (caveat + `corroboration_only` flag); a curated 16-class "incident signature" set gives the snippet-level indicator, audited against the GT snippets: **AP 0.6787 global / 0.6550 per-clip** on 500 annotated test clips (`data/features/aed/aed_stats.json`) - a correct corroboration signal, not an incident detector. Code: `safewatch/edge/aed.py`, `scripts/extract_aed_tags.py`, `scripts/check_vggish_space.py` (feature-space audit); 16 new tests |
| P6 | Supervision UI (Streamlit Console & Live Demo) | ✅ done · Interactive review console + live video upload, on-the-fly extraction & inference, and append-only audit trail (`app.py`). Run Review's causal EMA is display-only; Live Demo can optionally score growing prefixes (stride 8) so each prediction uses only past/current snippets. The upload is still decoded and features are extracted before scoring, so this is an offline causal simulation rather than a live frame-by-frame pipeline. **Live uploads only work with a checkpoint trained on live-reproducible features** (I3D visual + log-mel audio, `runs/2026-10-01_p3edge_i3dmel`): the on-the-fly extractor (`MiniRGB`, distilled from I3D) and log-mel patch stats reproduce *that* space only, so a VideoSwin/VGGish champion receives out-of-distribution inputs and scores ~0 on obvious incidents — the console now detects and warns about this (`state.live_feature_support`), and benchmark clips are scored from their real precomputed features (parity with `predictions.npz`) |
| P7 | Edge AI (ONNX fp32, latency/memory budget, Jetson/RPi export) | ✅ done · Complete pipeline benchmarked (**18× to 34× faster than real time**, total model footprint 5.67 MB, 6.5 ms fusion latency), deployment guide in `docs/edge_ai.md` |
| P8 | Robustness report | ✅ done · Consolidated synthesis and degradation tables in `docs/robustness_report.md` |

## Quality gates

```bash
.venv/bin/python -m pytest -q      # 107 tests: metrics (14) + pipeline (41) + explain (20) + edge (10) + analysis (8) + UI (14)
.venv/bin/python -m ruff check .   # clean
bash scripts/status.sh             # data/jobs/runs dashboard (jobs + val/test AP per run)
bash scripts/resume_jobs.sh        # after a reboot/crash: re-launch the queue if nothing runs
.venv/bin/python scripts/fusion_table.py        # ablation ladder + modality-removal columns
.venv/bin/python scripts/robustness_table.py     # P8 degradation sweeps, clean->worst
.venv/bin/python scripts/seed_margin.py --margin A B  # is a margin bigger than the seed spread?
.venv/bin/python scripts/ladder_seed_check.py    # ladder with seed ranges (which rows are replicated?)
bash scripts/snapshot.sh           # code+docs+lists tar -> ~/safewatch_snapshots (no Git, by choice)
```


## Hardware / constraints (measured, not assumed)

- **No CUDA GPU.** Mini-PC: AMD Ryzen 5 8640U (6C/12T), 14 GB RAM, ~206 GB free disk.
- PyTorch runs CPU-only; a representative fusion-head training step costs ~24 ms
  ⇒ local CPU training of feature-based models is viable (see `docs/journal.md`).
- Consequence: the **benchmark track uses precomputed features** (I3D RGB, VGGish audio), while the
  **edge/demo track** uses a light-weight extractor (ONNX, int8) sharing the same fusion head.

## Layout

```
data/       raw/ (downloads, gitignored)  features/  lists/  lists_partial/  lists_official/
            annotations/
safewatch/  data/ (dataset, multimodal, reliability)  eval/  losses/  models/
            train.py  train_fusion.py     # package, one phase at a time
scripts/    run_p1.sh  run_p1_official.sh  run_p2_audio.sh  run_p3_full.sh  run_p3_score.sh
            run_p3_clipnorm.sh (calibration fix)  run_p3_catchup.sh (queue)
            resume_jobs.sh (re-launch the queue after a reboot/crash)
            build_lists.py  extract_audio_features.py  audio_stats.py  quality_stats.py
            audit_audio_features.py (audio file/grid audit; JSON report)
            check_feature_grid.py  download_hf.py  robust_download.py
            eval_source_disjoint.py (leakage audit)  fusion_table.py (ladder + robustness table)
            per_class_localization.py (per-category incidents)  class_diagnostic.py (weak-class diagnosis)
            robustness_sweep.py (P8 degradation curves)  robustness_table.py (sweep summary)
            seed_margin.py (seed spread + paired margin: is a difference bigger than the seed noise?)
            ladder_seed_check.py (ladder with seed ranges: which rows are replicated?)
            run_causal_score.sh (P4 streaming/causal scoring of the reference models)
            fit_calibration.py (P5 alert confidence: global + --per-class temperature, val->test)
            export_onnx.py (P7 Edge: ONNX + int8 + latency/size budget)
            status.sh (dashboard)  snapshot.sh (no-Git safety net)  smoke_test.py
notebooks/  exploratory analysis
tests/      test_metrics.py (evaluation maths)  test_pipeline.py (data/model/reliability/contracts)
docs/       journal.md (decision log)  dataset.md (generated data facts)
            protocol.md (data/metric/method rules + comparability)
            theory_notes.md (T1-T10 theory you need)  reading_plan.md (what to read per phase)
            avancement.md (meeting/jury brief)  report/
export/     onnx/ (P7 artefacts: <run>_{fp32,int8}.onnx + <run>_report.json)
runs/       <date>_<experiment>/config.json + history.csv + metrics.json + ckpt_best.pt
            (+ predictions.npz + test_metrics.json once scored)
            (acts as the experiment history instead of Git)
data/logs/  one log per long job (p1_official, p2_audio, p3_score, p3_fusion, ...)
```

Fusion training and audio-based evaluation print an audio preflight summary before running,
including clips whose saved RMS features indicate digital silence. For a full per-clip audit report,
run `scripts/audit_audio_features.py` with the matching list CSVs, feature directory, audio grid,
snippet stride, and expected feature width; examples are in the script help and the latest workspace
audit is `sheets/audio_alignment_audit_20261008.md`.

No Git repository by choice: `docs/journal.md` + per-run `runs/<date>_<exp>/` snapshots are the
reproducibility record, and `scripts/snapshot.sh` keeps a rolling tar of code+docs+lists as a safety net.

## Environment

```bash
cd /home/aarf101/projtutore
python3 -m venv .venv
.venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py      # verifies imports, ffmpeg, CPU training step
```

## Data

Source: HuggingFace mirror `jherng/xd-violence` (MIT-tagged mirror of XD-Violence).

```bash
.venv/bin/python scripts/download_hf.py --list-videos --limit 10   # inspect
.venv/bin/python scripts/download_hf.py --all --sample-videos 3    # bootstrap
```

Formats, snippet stride, audio layout and ground-truth construction are documented in
`docs/dataset.md`. The dataset is research-only: **never redistribute** videos/features.

## References

- P. Wu et al., *Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak
  Supervision*, ECCV 2020 — https://roc-ng.github.io/XD-Violence
- P. Wu et al., *AVadCLIP: Audio-Visual Collaboration for Robust Video Anomaly Detection*, arXiv 2504.04495
- Y. Wang et al., *Multimodal Anomaly Detection in Complex Environments Using Video and Audio Fusion*,
  Scientific Reports 15:16291, 2025
