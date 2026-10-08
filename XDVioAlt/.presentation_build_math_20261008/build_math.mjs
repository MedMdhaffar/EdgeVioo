import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = "/home/aarf101/projtutore";
const skillDir = "/home/aarf101/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations";
const workDir = path.join(root, ".presentation_build_math_20261008");
const finalPptx = path.join(root, "presentations", "SafeWatch_Model_Math_20261008.pptx");
const font = "DejaVu Sans";
const C = {
  bg: "#0A1220", ink: "#F4F7FB", muted: "#B4C0D2", quiet: "#8291A7",
  cyan: "#56D7CB", amber: "#F4C66A", blue: "#7DAEFF", pale: "#DCE7F5",
  plot: "#101C2D", grid: "#34445B", danger: "#F28B82",
};
const ppt = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function text(slide, value, x, y, w, h, size = 24, color = C.ink, bold = false,
              name = undefined) {
  const s = slide.shapes.add({
    geometry: "textbox", name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0 },
  });
  s.text = value;
  s.text.style = {
    typeface: font, fontSize: size, color, bold, verticalAlignment: "middle",
    autoFit: "shrinkText", wrap: true,
  };
  return s;
}

function newSlide(title, n, subtitle = "") {
  const slide = ppt.slides.add();
  slide.background.fill = C.bg;
  text(slide, title, 72, 42, 1136, 58, 42, C.ink, true, `slide-${n}-title`);
  if (subtitle) text(slide, subtitle, 74, 106, 1128, 38, 21, C.muted, false);
  text(slide, `SAFEWATCH  /  MODEL MATHEMATICS                                      ${String(n).padStart(2, "0")}`,
       74, 676, 1132, 20, 14, C.quiet, false);
  return slide;
}

// 1. Title
{
  const s = ppt.slides.add();
  s.background.fill = C.bg;
  text(s, "SafeWatch Edge AI", 76, 110, 1128, 78, 56, C.ink, true);
  text(s, "Mathematics of the audiovisual detector", 78, 202, 1100, 54, 34, C.cyan, false);
  text(s, "The p3full_cross benchmark model", 80, 278, 960, 38, 24, C.muted, false);
  text(s, "V, A  →  temporal encoders  →  cross-attention  →  snippet and clip scores",
       80, 410, 1118, 66, 27, C.pale, false);
  text(s, "VideoSwin 768-d  ·  log-mel 132-d  ·  weak clip-level supervision",
       82, 502, 1080, 38, 22, C.quiet, false);
  text(s, "SAFEWATCH  /  08 OCT 2026", 82, 650, 900, 20, 14, C.quiet, false);
  s.speakerNotes.textFrame.setText(
    "Introduce the exact checkpoint: runs/2026-09-29_p3full_cross. Explain that this deck follows the math from feature construction through training and inference. The benchmark model consumes precomputed features; it does not train the Swin extractor end-to-end.");
}

// 2. Feature construction and synchronization
{
  const s = newSlide("Feature construction and alignment", 2,
    "Each training example becomes two sequences with the same time index t.");
  text(s, "VIDEO", 78, 160, 190, 28, 18, C.cyan, true);
  text(s, "V = [v₁,…,vT] ∈ ℝᵀˣ⁷⁶⁸", 78, 195, 520, 42, 27, C.ink, true);
  text(s, "Frozen Swin descriptor per video snippet", 80, 244, 535, 30, 20, C.muted);

  text(s, "AUDIO", 680, 160, 190, 28, 18, C.amber, true);
  text(s, "Xₘₖ = Σₙ₌₀³⁹⁹ x[n+160m]w[n]e⁻ʲ²πkn/400", 680, 194, 520, 50, 22, C.ink, true);
  text(s, "Pᵦₘ = Σₖ Hᵦ[k]|Xₘₖ|²", 680, 250, 520, 34, 23, C.pale);
  text(s, "Mᵦₘ = 10 log₁₀(Pᵦₘ / Pᵣₑ𝒻)", 680, 289, 520, 34, 23, C.pale);
  text(s, "64 mel bands · 96 frames per patch · 0.96 s", 682, 335, 510, 30, 19, C.muted);

  text(s, "μᵦ = (1/96) Σₘ Mᵦₘ", 80, 405, 495, 40, 25, C.ink, true);
  text(s, "σᵦ = √[(1/96) Σₘ (Mᵦₘ − μᵦ)²]", 80, 451, 530, 42, 24, C.ink, true);
  text(s, "64 means + 64 standard deviations + 4 extra slots = 132 features", 80, 535, 1108, 34,
       22, C.cyan, true);

  text(s, "Δtᵥ = 64/24 = 2.67 s", 680, 405, 520, 36, 24, C.amber, true);
  text(s, "patches per video snippet = 2.67/0.96 ≈ 2.78", 680, 450, 520, 36, 22, C.ink);
  text(s, "aₜ = (1/|Pₜ|) Σₚ∈Pₜ aₚ", 680, 493, 520, 40, 24, C.ink, true);
  text(s, "âₜⱼ = (aₜⱼ − μⱼ(train)) / max(σⱼ(train), 10⁻⁶)", 80, 588, 1120, 34,
       22, C.pale);
  s.speakerNotes.textFrame.setText([
    "Video features are precomputed Swin vectors with dimension 768 for each snippet. The visual extractor is frozen for this benchmark head.",
    "Audio is decoded to 16 kHz mono. The STFT uses N=400 samples (25 ms) and hop H=160 samples (10 ms). H_b is the b-th mel filter. The code uses 64 mel bins and power-to-decibel conversion relative to the clip maximum.",
    "The 132-d vector contains 64 per-band means and 64 per-band standard deviations plus four extra slots. Implementation caveat: flux is reduced to a clip-level average and repeated over patches; one extra slot is zero. Avoid claiming four independent patch-wise descriptors.",
    "Audio patches are pooled onto the 64-frame video grid. Training-set means and standard deviations are used for feature normalization. References: scripts/extract_audio_features.py, safewatch/data/multimodal.py.",
  ]);
}

// 3. Temporal encoders
{
  const s = newSlide("Temporal encoders", 3,
    "Separate convolution stacks map both modalities into a shared 128-dimensional space.");
  text(s, "Hᵥ = fᵥ(V) ∈ ℝᵀˣ¹²⁸", 82, 173, 510, 44, 28, C.cyan, true);
  text(s, "Hₐ = fₐ(A) ∈ ℝᵀˣ¹²⁸", 682, 173, 510, 44, 28, C.amber, true);
  text(s, "For either stream xₜ:", 82, 247, 440, 30, 21, C.muted, false);
  text(s, "uₜ⁽¹⁾ = ReLU(W₁xₜ + b₁)             d → 512", 82, 294, 1100, 43,
       25, C.ink, true);
  text(s, "uₜ⁽²⁾ = ReLU(W₂uₜ⁽¹⁾ + b₂)        512 → 128", 82, 355, 1100, 43,
       25, C.ink, true);
  text(s, "hₜ = ReLU(b₃ + Σᵣ₌₋₂² W₃,ᵣ uₜ₊ᵣ⁽²⁾)      Conv1D, kernel = 5", 82, 416,
       1120, 52, 24, C.ink, true);
  text(s, "Receptive field: five neighboring snippets", 82, 489, 700, 32,
       21, C.cyan, true);
  text(s, "Dropout p = 0.6 between channel-mixing layers during training", 82, 541,
       1050, 32, 20, C.muted);
  text(s, "Convolutions learn local temporal patterns; attention later connects distant snippets.",
       82, 592, 1110, 44, 22, C.pale);
  s.speakerNotes.textFrame.setText([
    "The exact encoder is Conv1d(input,512,kernel=1), ReLU, Dropout(0.6), Conv1d(512,128,kernel=1), ReLU, Dropout(0.6), Conv1d(128,128,kernel=5,padding=2), ReLU.",
    "The first two layers mix channels at each time step. The final layer mixes five neighboring time steps. Padding is symmetric, so this offline feature encoder can use neighboring future snippets near a time point.",
    "The video and audio encoders are separate; each emits 128 dimensions per snippet.",
  ]);
}

// 4. Bidirectional cross-attention
{
  const s = newSlide("Bidirectional cross-attention", 4,
    "Each modality queries the other; four heads operate in parallel.");
  text(s, "One head: video queries audio", 80, 165, 560, 30, 21, C.cyan, true);
  text(s, "Qᵥ = HᵥWQ,   Kₐ = HₐWK,   Uₐ = HₐWV", 80, 210, 1120, 42,
       26, C.ink, true);
  text(s, "Cᵥ←ₐ = softmax(QᵥKₐᵀ / √dₖ) Uₐ", 80, 270, 1120, 48,
       30, C.ink, true);
  text(s, "dₖ = 128/4 = 32", 82, 329, 430, 36, 24, C.amber, true);
  text(s, "Attention map: T × T; each video snippet weights all valid audio snippets.",
       82, 373, 1100, 36, 21, C.muted);
  text(s, "Cₐ←ᵥ = softmax(QₐKᵥᵀ / √32) Uᵥ", 80, 436, 1120, 44,
       27, C.pale, true);
  text(s, "H′ᵥ = LayerNorm(Hᵥ + Cᵥ←ₐ)     H′ₐ = LayerNorm(Hₐ + Cₐ←ᵥ)",
       80, 491, 1120, 46, 23, C.ink);
  text(s, "hₜ = Wₘ[H′ᵥ,ₜ ; H′ₐ,ₜ] + bₘ     256 → 128", 80, 557, 1120, 44,
       24, C.cyan, true);
  s.speakerNotes.textFrame.setText([
    "Q, K and V are learned linear projections. The scaled dot product gives pairwise temporal compatibility. Dividing by sqrt(d_k) keeps dot-product magnitudes controlled before softmax.",
    "This implementation has two MultiheadAttention modules: visual queries audio, and audio queries visual. Each uses four heads of width 32, residual addition and LayerNorm.",
    "Reliability nuance: the first reliability channel scales the audio keys and values used in the visual-query branch by sigmoid(r). The reverse branch uses the original audio embeddings. This is not the learned alpha gate in the adaptive variant.",
    "The attention mechanism follows the standard Transformer scaled dot-product formula (Vaswani et al., 2017, https://arxiv.org/abs/1706.03762). The project code is safewatch/models/fusion.py.",
  ]);
}

// 5. Pooling and output heads
{
  const s = newSlide("Temporal pooling and output heads", 5,
    "The fused sequence produces both snippet-level and clip-level predictions.");
  text(s, "Snippet anomaly probability", 80, 165, 480, 30, 20, C.cyan, true);
  text(s, "zₜ = wₛᵀhₜ + bₛ", 80, 204, 500, 40, 27, C.ink, true);
  text(s, "pₜ = σ(zₜ) = 1/(1 + e⁻ᶻᵗ)", 80, 254, 520, 42, 26, C.ink);

  text(s, "Attention pooling", 680, 165, 500, 30, 20, C.amber, true);
  text(s, "eₜ = wₐᵀhₜ + bₐ", 680, 204, 520, 39, 25, C.ink, true);
  text(s, "αₜ = exp(eₜ) / Σᵤ exp(eᵤ)", 680, 250, 520, 42, 24, C.ink);
  text(s, "hclip = Σₜ αₜhₜ", 680, 300, 520, 42, 26, C.ink, true);

  text(s, "Clip anomaly", 80, 390, 320, 30, 20, C.cyan, true);
  text(s, "pclip = σ(wᵧᵀhclip + bᵧ)", 80, 431, 500, 43, 25, C.ink, true);
  text(s, "Six category probabilities", 680, 390, 500, 30, 20, C.amber, true);
  text(s, "q = σ(Wc hclip + bc) ∈ (0,1)⁶", 680, 431, 520, 43,
       24, C.ink, true);

  text(s, "Categories use independent sigmoid outputs, so one clip may have multiple labels.",
       80, 534, 1110, 38, 22, C.pale);
  text(s, "Masked softmax excludes padded snippets from αₜ.", 80, 584, 1000, 32,
       20, C.muted);
  s.speakerNotes.textFrame.setText([
    "The snippet head is a 1x1 Conv1d from 128 dimensions to one logit. Its sigmoid score is the timeline value used by the evaluator.",
    "The clip attention scorer is a learned linear score over h_t, followed by masked softmax. The pooled representation is a weighted sum. One linear head yields clip anomaly; another yields six multi-label logits.",
    "The category order in the code is fighting, shooting, riot, abuse, car_accident, explosion. Six sigmoids are used rather than a softmax because labels can co-occur.",
    "Attention weights are part of the model computation and can help visualize which snippets were weighted. They are not, by themselves, proof that the model used a human-interpretable causal reason.",
  ]);
}

// 6. Weak-label training objective
{
  const s = newSlide("Weak-label training objective", 6,
    "Clip labels supervise ranking across time; the model does not receive exact boundary labels.");
  text(s, "Top-5 multiple-instance score", 80, 159, 610, 29, 20, C.cyan, true);
  text(s, "pbag = (1/5) Σₜ∈Top5(p) pₜ", 80, 198, 1090, 44, 28, C.ink, true);
  text(s, "LMIL = −y log(pbag) − (1−y) log(1−pbag)", 80, 251, 1090, 46,
       25, C.pale);

  text(s, "Attention clip loss", 80, 330, 500, 28, 20, C.amber, true);
  text(s, "Latt = BCE(pclip, y)", 80, 366, 520, 38, 24, C.ink, true);
  text(s, "Multi-label category loss", 680, 330, 520, 28, 20, C.amber, true);
  text(s, "Lmulti = −(1/6) Σc [yc log(qc) + (1−yc) log(1−qc)]",
       680, 366, 520, 68, 20, C.ink, true);

  text(s, "L = LMIL + 0.5 Latt + 0.3 Lmulti", 80, 474, 1110, 52,
       31, C.cyan, true);
  text(s, "Adam · learning rate 10⁻⁴ · 30 epochs · batch 32", 82, 543, 1000, 34,
       21, C.pale);
  text(s, "Modality dropout: p = 0.15 during training", 82, 590, 1000, 32,
       20, C.muted);
  s.speakerNotes.textFrame.setText([
    "For each clip, pbag is the mean of the five largest valid snippet probabilities. k_eff is reduced for clips with fewer than five valid snippets.",
    "LMIL is binary cross-entropy against the clip anomaly label. Latt is binary cross-entropy on the separate attention-pooled clip score. Lmulti is mean binary cross-entropy across six independent category outputs and the batch. This checkpoint uses no class positive weighting.",
    "The configured total is LMIL + 0.5*Latt + 0.3*Lmulti. Training uses Adam, initial learning rate 1e-4, batch size 32, 30 epochs, and a learning-rate milestone at epoch 10 with gamma 0.1. The config uses 15% modality dropout.",
    "The top-five surrogate assumes a positive clip should have some high-scoring snippets. It does not say which precise frames start or end the event. This follows the weak-label setting of Wu et al., XD-Violence, ECCV 2020 (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750324.pdf), while the model architecture is project-specific.",
  ]);
}

// 7. Inference, evaluation and limits
{
  const s = newSlide("Inference, evaluation, and limits", 7,
    "Snippet probabilities become intervals only after thresholding and temporal grouping.");
  const labels = ["Fighting", "Shooting", "Riot", "Abuse", "Car accident", "Explosion"];
  const values = [0.8408, 0.5374, 0.9552, 0.0171, 0.9065, 0.7662];
  const chart = s.charts.add("bar", {
    position: { left: 52, top: 170, width: 690, height: 416 },
    categories: labels,
    series: [{ name: "Per-class AP", values, fill: C.cyan }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 56 },
    hasLegend: false,
    chartFill: C.bg,
    chartLine: { fill: C.bg, width: 0 },
    plotAreaFill: C.bg,
    plotAreaLine: { fill: C.bg, width: 0 },
    xAxis: { min: 0, max: 1, majorUnit: 0.2, numberFormatCode: "0.0", textStyle: { typeface: font, fontSize: 16, fill: C.muted }, majorGridlines: { style: "solid", fill: C.grid, width: 1 } },
    yAxis: { textStyle: { typeface: font, fontSize: 16, fill: C.ink }, majorGridlines: null },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: font, fontSize: 15, fill: C.ink, bold: true } },
  });
  const { applyPresentationChartFont } = await import(
    pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
  applyPresentationChartFont(chart, { fontFamily: font });

  text(s, "pₜ ≥ 0.5", 786, 174, 400, 40, 28, C.cyan, true);
  text(s, "Join detections across gaps ≤ 1 snippet", 786, 224, 414, 52,
       21, C.ink);
  text(s, "Require ≥ 2 snippets", 786, 284, 414, 40, 21, C.ink);
  text(s, "64-frame grid: 1 snippet ≈ 2.67 s", 786, 342, 414, 48,
       20, C.muted);
  text(s, "Test metrics", 786, 420, 400, 28, 19, C.amber, true);
  text(s, "Global PR-AUC  0.7981", 786, 456, 410, 34, 23, C.ink, true);
  text(s, "Per-video AP  0.7613", 786, 496, 410, 34, 23, C.ink, true);
  text(s, "Intervals are post-processing; exact boundaries are not trained.",
       786, 554, 410, 68, 19, C.pale);
  text(s, "Class AP shown from this run; abuse has 11 positive test clips.",
       80, 612, 1100, 30, 18, C.muted);
  s.speakerNotes.textFrame.setText([
    "The chart uses per_class_clip_level_ap from runs/2026-09-29_p3full_cross/test_metrics.json: fighting .8408, shooting .5374, riot .9552, abuse .0171, car accident .9065, explosion .7662. Abuse has 11 positives in this test list, so its estimate is based on a small sample.",
    "The same test_metrics file reports global__pr_auc=.798119, global__average_precision=.797052, and per_video__average_precision=.761273. These are different averaging protocols; do not label all of them simply 'AP' without specifying which.",
    "The saved segment rule uses threshold 0.5, min_length=2 snippets, max_gap=1 snippet. At 64/24 seconds per snippet that becomes a 5.33-second minimum duration and a 2.67-second gap.",
    "Localization segments are derived from scores after model inference. They are not outputs of a learned boundary regressor and training did not use temporal interval annotations.",
  ]);
}

await fs.mkdir(workDir, { recursive: true });
await fs.mkdir(path.dirname(finalPptx), { recursive: true });
const draft = path.join(workDir, "safewatch_math_draft.pptx");
await (await PresentationFile.exportPptx(ppt)).save(draft);
for (let i = 0; i < ppt.slides.items.length; i += 1) {
  const slide = ppt.slides.items[i];
  const png = await ppt.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(workDir, `slide-${i + 1}.png`),
    new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(workDir, `slide-${i + 1}.layout.json`), await layout.text());
}
console.log(JSON.stringify({ draft, finalPptx, slides: ppt.slides.items.length, font }));
