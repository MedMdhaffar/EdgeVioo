# Reading plan — what to read before each phase

> Rule: read the **minimum** that unblocks the next phase, then stop at the "stop when" criterion.
> Papers are listed with the sections worth your time; the rest of each paper is skippable.
> Level key (see `docs/theory_notes.md`): **A** implement · **B** use · **C** cite only.

## How to read a paper here (30–60 min per pass)

1. Abstract + figure captions + conclusion (5 min) — what problem, what trick, what result.
2. The one "how do I build it" section (20 min) — take notes on **shapes and losses**, not prose.
3. The ablation tables (15 min) — which component actually buys the improvement?
4. Open the reference implementation and locate that component in code (20 min).

---

## Before P1 — visual-only baseline (CPU, precomputed I3D features)

| What | Why | Stop when you can… | Level | Time |
|---|---|---|---|---|
| **Our harness**: `safewatch/eval/metrics.py` + `tests/test_metrics.py` | the evaluation protocol is the highest-value theory in this project | explain `pr_auc` vs `average_precision`, and why tied scores give 0.50 vs 0.75 (we measured this) | A | 1 h |
| sklearn docs: `precision_recall_curve`, `average_precision_score`, `auc` | these are the exact functions the official code calls in `test.py` | state what `auc(recall, precision)` integrates, and where ties break it | A | 30 min |
| **Sultani et al. 2018**, *Real-world Anomaly Detection in Surveillance Videos*, CVPR — MIL formulation | origin of the MIL objective every WSVAD paper inherits | write the MIL objective with sparsity/diversity constraints | A | 1.5 h |
| **HL-Net / XD-Violence, ECCV 2020** — arXiv:2007.04687: abstract, Method (three branches + approximator), the multimodal ablation | the dataset's reference baseline *and* the source of the AP protocol | describe holistic/localized/score branches and quantify how much audio adds | A | 2.5 h |
| `XDVioDet` reference code: `dataset.py`, `model.py`, `train.py`, `test.py`, `option.py` | see MIL + top-k + the 16-frame evaluation hack in ~400 lines | point to the line that averages the 5 crops and the line that repeats predictions over frames | B | 1.5 h |
| `docs/dataset.md` (ours) | the stride/annotation facts that change the protocol | explain why we must use a measured ~63-frame stride instead of 16 | A | 20 min |

**Deliverable of this reading block:** a half-page note in `docs/journal.md` answering
"what is our baseline, what objective does it optimise, and how will we score it?".

---

## Before P2 — audio pipeline (spectrograms, VGGish, event tagging, signal quality)

| What | Why | Stop when you can… | Level | Time |
|---|---|---|---|---|
| **VGGish** (Hershey et al. 2017, ICASSP, *CNN Architectures for Large-Scale Audio Classification*) + the `tensorflow/models/research/audioset/vggish` README parameters | tells you exactly what the 128-d audio embedding is and its input contract | state the input contract (16 kHz mono, 0.96 s patch, 64-mel, 96 frames) → output shape | B | 1 h |
| Log-mel spectrogram basics (librosa docs: `stft`, `melspectrogram`, `power_to_db`) | you must *plot* and *explain* spectrograms in the report and in the UI | justify `n_fft` / `hop_length` / `n_mels` choices; read a real mel plot | B | 1 h |
| **PANNs** (Kong et al. 2020, TASLP) + `panns-inference` README (AudioSet 527 classes, 0.32 s frame hop) | source of the "indices sonores" phrases in alerts and of abnormal-sound segments | list which AudioSet classes you will map to the French phrases | C | 1 h |
| Our ffmpeg recipe (downmix 6ch → mono, 48 kHz → 16 kHz) | the real data is 48 kHz **6-channel** AAC — verified in `docs/dataset.md` | write the exact ffmpeg filter chain and explain each part | A | 30 min |
| Signal-quality estimation (RMS energy, spectral flatness, zero-crossing rate, VAD activity) | the subject explicitly requires *estimation de la qualité du signal*; the adaptive gate consumes it | define a per-snippet `audio_reliability` feature | B | 1 h |

**Deliverable:** `data/features/vggish/` populated for the 7 sample clips + one mel-spectrogram
figure (normal vs explosion vs fighting) + a documented `audio_reliability` formula.

---

## Before P3 — multimodal fusion (the technical core)

| What | Why | Stop when you can… | Level | Time |
|---|---|---|---|---|
| **Vaswani et al. 2017**, *Attention Is All You Need* — §3.1 scaled dot-product, §3.2 multi-head | cross-attention is the heart of the fusion module | write `Attention(Q,K,V) = softmax(QKᵀ/√d)V` and give the tensor shapes for our case | A | 2 h |
| **CMA** (ICCECE 2022, *Audio-Guided Attention Network*) + code `yujiangpu20/cma_xdVioDet/model.py` | the smallest complete cross-modal attention implementation for XD-Violence | trace every tensor shape through that file, end to end | A | 2 h |
| **AVadCLIP** arXiv:2504.04495 — §III-C *Audio-Visual Adaptive Fusion*, §III-F *Uncertainty-Driven Distillation*, §IV-D1 and §IV-D3 ablations | this is the "adaptive fusion" + "works when a modality degrades" requirement, formally described | explain how the adaptive gate is computed and what distillation buys when audio is missing | A | 2 h |
| HL-Net multimodal ablation (RGB / RGB+audio / RGB+flow / all) | quantify how much audio actually adds (and costs) | quote the AP deltas and their cost in features | C | 20 min |
| Wang et al. 2025, *Sci Rep* 15:16291 (STADNet) — architecture + cross-attention fusion | third pré-requis: cross-attention framing, multi-scale 3D conv | describe it **and** state the differences (no audio fusion, Ped2/Avenue, frame-level supervised AUC) — cite for framing only | C | 1 h |
| Class-imbalance handling for the 6 categories (`abuse` has only 8 test clips) | decides loss weighting and per-class thresholds | pick a strategy and justify it | B | 30 min |

**Deliverable:** the fusion ablation matrix (early / late / cross-attention / adaptive-gated), each row
with AP (global + per-video), per-class AP, and the degraded-modality columns.

---

## Before P4 — temporal localisation (offline + online)

| What | Why | Stop when you can… | Level | Time |
|---|---|---|---|---|
| HL-Net *online* detection (the `approximator` branch) + its online-vs-offline AP comparison | "temps quasi réel" means causal scoring; the reference shows what that costs | explain the approximator and quote the offline→online AP drop (0.7864 → 0.7553 in the leaderboard) | A | 1 h |
| Class-specific temporal scores from weak labels — skim one of MSBT / UR-DMU / MGFN (the class-attention block only) | turns one anomaly score into the 6 per-category score curves the UI needs | sketch `A_class = softmax(Linear(CAT(x)) ⊙ Linear(x))` on paper | B | 1.5 h |
| Our `localization_report` + choosing threshold / `min_length` / `max_gap` on the **validation** split | converts a score curve into alert segments | justify the three parameters from the validation PR curve, not by eye | A | 1 h |
| How the literature scores localisation (point-based vs segment-based; skim VadCLIP/UR-DMU eval code) | avoid inventing a metric nobody uses | state which metric we report and why | C | 1 h |

**Deliverable:** `docs/localization.md` — the segment-extraction rule, the parameter values and their
validation justification, plus the metric table.

---

## Before P5 — explanations & operator-facing evidence

| What | Why | Stop when you can… | Level | Time |
|---|---|---|---|---|
| **Grad-CAM** (Selvaraju et al. 2017) — §1–2 only | "images ou zones significatives" for the light (edge) visual track | explain the class-weighted activation map in three lines | B | 1 h |
| **"Attention is not Explanation"** (Jain & Wallace 2019) | protects you from over-claiming in the report/report defence | phrase the caveat correctly: attention = *evidence*, not *proof* | C | 30 min |
| Leave-one-modality-out contribution (our own design) | the subject requires "contribution de chaque modalité" | write the formula (`Δ = score_both − score_without_modality`) and note its cost | A | 30 min |
| AVadCLIP §IV-E qualitative results | how strong papers present evidence figures | design our alert figure (timeline + frames + mel) | C | 30 min |
| French phrasing discipline for alerts | the fiche must read like the subject's example, and must not claim more than the cues support | draft the template with the AudioSet→French mapping | B | 1 h |

---

## Before P6 — supervision UI

| What | Why | Stop when you can… | Level | Time |
|---|---|---|---|---|
| Streamlit docs: `st.video(start_time=, end_time=)`, `st.audio`, `st.plotly_chart`, `session_state`, `components.v1.html` | the whole UI stack (already installed, v1.64) | verify segment-jump playback works with a local mp4 | B | 1.5 h |
| Plotly: `add_vrect`, subplots with shared x-axis, hover | the probability timeline + audio/contribution tracks | build the timeline figure in a notebook before touching the app | B | 1.5 h |
| ffmpeg docs: `rtsp` input + `hls` muxer | live mode and browser playback (browsers cannot play RTSP) | explain the RTSP→HLS chain and where the latency comes from | C | 1 h |
| `sqlite3` (stdlib) — CREATE TABLE, transactions | alert store + audit log with no new dependency | sketch the schema: alerts / decisions / runs | B | 30 min |
| HTML→PDF (browser print) vs `reportlab` | fiche d'incident export | decide and justify (zero new deps vs nicer PDFs) | C | 30 min |

---

## Before P7 — Edge AI (optimisation & constrained deployment)

| What | Why | Stop when you can… | Level | Time |
|---|---|---|---|---|
| ONNX Runtime quantisation docs: dynamic vs static (QDQ), per-channel, INT8 | "une version optimisée du modèle" is a graded requirement | explain why dynamic quantisation needs no calibration data and why convs quantise better than attention | B | 1.5 h |
| `onnx` 1.23 + `onnxruntime` 1.30 API: `quantize_dynamic`, providers, graph optimisations | you will actually run this; both are already installed | produce a quantised model and measure size + latency yourself | A | 1.5 h |
| TensorRT / Jetson guides (skim `trtexec`, INT8 calibration, DLA) | we ship **untested** Jetson export scripts — know exactly what they assume | describe the export path and its prerequisites, and say what is unverified | C | 1 h |
| Docker resource limits (`--cpus`, `--memory`) | simulates the constrained platform with reproducible numbers | run our pipeline inside a 2-CPU/1 GB envelope and record the outcome | B | 1 h |

**Deliverable:** `docs/edge_deployment.md` — the latency/RAM/size table (fp32 vs int8), the constrained
profile results, and an explicit "verified / not verified without hardware" column.

---

## Before P8 — report & defence

| What | Why | Stop when you can… | Level | Time |
|---|---|---|---|---|
| Report skeleton (contexte → état de l'art → méthode → expériences → limites → conclusion) | the report is itself a deliverable | outline it using only our own documents as sources | B | 1 h |
| Our extracted leaderboard numbers + the comparability caveat (2048-d/64-frame vs official 1024-d/16-frame features) | never compare apples with oranges in a graded report | state which numbers are ours (re-baselined) and which are published | A | 30 min |
| `docs/journal.md` + `runs/<date>_<exp>/config.yaml` snapshots | reproducibility is a claimed property — it must be true | reproduce one past run using only the journal | A | 1 h |
| The three traps (metrics ties, stride/GT alignment, leakage) | they are your strongest "I understand evaluation" talking point at the defence | explain each in 60 s with our measured numbers | A | 30 min |

---

## Appendix — full reference list with links

| Ref | Used for | Where |
|---|---|---|
| Wu et al., *Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision*, ECCV 2020 — arXiv:2007.04687 | dataset, baseline, AP protocol, three branches, online mode | P1, P4 |
| Code: `github.com/Roc-Ng/XDVioDet` (model/dataset/train/test/option) | reference implementation of MIL + top-k + evaluation | P1 |
| Dataset page: `roc-ng.github.io/XD-Violence` | official features/annotations, leaderboard context | P0 |
| Wu et al., *AVadCLIP: Audio-Visual Collaboration for Robust Video Anomaly Detection*, arXiv:2504.04495 (§III-C, §III-D, §III-F, §IV-D1/D3, §IV-E) | adaptive fusion, prompts, uncertainty-driven distillation, robustness ablations, qualitative figures | P3, P5 |
| Wang et al., *Multimodal Anomaly Detection in Complex Environments Using Video and Audio Fusion*, Sci Rep 15:16291 (2025) | cross-attention framing, multi-scale 3D conv — **different setting**, cite with care | P3 |
| `github.com/yujiangpu20/cma_xdVioDet` (audio-guided attention, ICCECE 2022) | smallest complete cross-modal attention implementation for XD-Violence | P3 |
| Sultani et al., *Real-world Anomaly Detection in Surveillance Videos*, CVPR 2018 | origin of the MIL formulation | P1 |
| Hershey et al., *CNN Architectures for Large-Scale Audio Classification*, ICASSP 2017 (VGGish) | the 128-d audio embedding and its input contract | P2 |
| Kong et al., *PANNs*, IEEE/ACM TASLP 2020 + `panns-inference` | AudioSet event tagging for alert wording | P2, P5 |
| Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 (§3.1–3.2) | cross-attention mechanics | P3 |
| Selvaraju et al., *Grad-CAM*, ICCV 2017 | spatial evidence in alerts | P5 |
| Jain & Wallace, *Attention is not Explanation*, NAACL 2019 | honest framing of attention-based evidence | P5, report |
| HuggingFace mirror `jherng/xd-violence` | our actual data source (videos + I3D features + test annotations) | P0 |
| ONNX Runtime quantisation docs | int8 export for the Edge deliverable | P7 |



