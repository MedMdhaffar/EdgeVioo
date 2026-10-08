#!/usr/bin/env python
"""P7 (Edge AI) - export a trained checkpoint to ONNX, quantise it to int8, and measure the budget.

The subject asks for a version "deployee ou simulee sur une plateforme a ressources limitees"
(mini-PC / Jetson / Raspberry Pi + accelerator) and for privacy-friendly edge processing. This
script produces the artefact and the measurements that make the claim checkable on *this* machine:

    export/onnx/<run>_fp32.onnx     float32 graph (dynamic time axis)
    export/onnx/<run>_int8.onnx     dynamically quantised int8 (MatMul weights)
    export/onnx/<run>_report.json   sizes, latency percentiles, fp32-vs-int8 parity

Honesty rules, because an Edge claim is easy to inflate:
* latency is measured on the CPU that is really available here (no GPU, no Jetson), so the report
  says `device: cpu/<thread count>` and never converts into a Jetson number;
* the int8 model is only accepted if the *snippet logits* stay within a stated tolerance of fp32
  (`max_abs_diff` in the report) - a quantised model that changes decisions silently is not a win;
* attention weights are intentionally **not** exported: the explanation panel needs them, so the
  demo track runs the PyTorch model for explanations and the ONNX graph only for scoring. That is
  stated in the report rather than pretended away.

Usage:
    .venv/bin/python scripts/export_onnx.py --ckpt runs/2026-09-28_p3full_early/ckpt_best.pt --int8
    .venv/bin/python scripts/export_onnx.py --ckpt runs/2026-09-23_p1_visual_swin/ckpt_best.pt \
        --bench 50 --threads 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

# `python scripts/export_onnx.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safewatch.data.multimodal import AUDIO_DIM  # noqa: E402
from safewatch.eval.runner import load_model  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Thin wrapper: fixed input order, only the two outputs the runtime needs.

    Handles both model kinds: the fusion head takes ``(visual, audio, mask, reliability)`` while the
    unimodal MIL baseline takes ``(features, mask)``; unused inputs are pruned by the tracer, so the
    deployed graph exposes only what it really consumes.

    ``snippet_logits`` drives the timeline and the alert segments, ``multi_logits`` the category.
    Attention weights and the gate value ``alpha`` are deliberately excluded - they are explanation
    artefacts produced by the PyTorch model, not by the deployed graph.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.fusion = bool(getattr(model, "mode", None))

    def forward(self, visual: torch.Tensor, audio: torch.Tensor, mask: torch.Tensor,
                reliability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fusion:
            out = self.model(visual, audio, mask, reliability)
        else:                                   # MILModel: one visual stream, no audio branch
            out = self.model(visual, mask)
        return out["snippet_logits"], out["multi_logits"]


def strip_attention_weights(model: torch.nn.Module) -> bool:
    """Make bidirectional cross-attention exportable by not requesting attention weights.

    ``need_weights=True`` forces torch's slow, hard-to-export attention path. The graph only needs
    the refined embeddings, so the cross module's forward is rebound here (in the export process).
    """
    cross = getattr(model, "cross", None)
    if cross is None:                       # early / late fusion have no cross-attention
        return False

    def forward(self, visual, audio, pad_mask=None, reliability=None):
        # Mirrors CrossModalAttention.forward (keep in lockstep): the only change is
        # need_weights=False, which is what makes the graph exportable.
        audio_in = audio
        if reliability is not None and reliability.shape[-1] >= 1:
            rel_scale = torch.sigmoid(reliability[:, :, :1]) if reliability.ndim == 3 else 1.0
            audio_in = audio * rel_scale
        v2 = self.v_from_a(visual, audio_in, audio_in, key_padding_mask=pad_mask,
                           need_weights=False)[0]
        a2 = self.a_from_v(audio, visual, visual, key_padding_mask=pad_mask,
                           need_weights=False)[0]
        return self.norm_v(visual + v2), self.norm_a(audio + a2), None, None

    cross.forward = types.MethodType(forward, cross)
    return True


def model_shape(model: torch.nn.Module, cfg: dict) -> tuple[int, int, int]:
    """``(visual_dim, audio_dim, n_reliability)`` read from the weights, config as fallback.

    Early fusion concatenates both modalities into a *single* encoder, so the split cannot be read
    off that encoder - the checkpoint config carries ``visual_dim``/``audio_dim`` for exactly this
    reason, and the concatenated width is checked against them instead of being trusted.
    """
    if getattr(model, "mode", None):        # FusionModel
        state = model.state_dict()
        visual_weight = state.get("visual_encoder.net.0.weight")
        audio_weight = state.get("audio_encoder.net.0.weight")
        if visual_weight is not None and audio_weight is not None:
            visual_dim, audio_dim = int(visual_weight.shape[1]), int(audio_weight.shape[1])
        else:                               # early fusion: encoder over concat(visual, audio)
            visual_dim = int(cfg.get("visual_dim", 768))
            audio_dim = int(cfg.get("audio_dim", AUDIO_DIM))
            width = int(state["encoder.net.0.weight"].shape[1])
            if width != visual_dim + audio_dim:
                raise SystemExit(f"[onnx] early-fusion width {width} != visual {visual_dim} + "
                                 f"audio {audio_dim}: the config does not match the weights")
        return visual_dim, audio_dim, int(getattr(model, "n_reliability", 0))
    in_dim = int(getattr(model, "in_dim", cfg.get("in_dim", 768)))
    return in_dim, 0, 0


def make_inputs(visual_dim: int, audio_dim: int, n_reliability: int, n_snippets: int,
                batch: int = 1) -> tuple[torch.Tensor, ...]:
    """Realistic dummy batch: all snippets valid (a padded mask is the exception, not the norm)."""
    visual = torch.randn(batch, n_snippets, visual_dim)
    audio = (torch.randn(batch, n_snippets, audio_dim) if audio_dim
             else torch.zeros(batch, n_snippets, 1))
    mask = torch.ones(batch, n_snippets, dtype=torch.bool)
    reliability = (torch.randn(batch, n_snippets, n_reliability) if n_reliability
                   else torch.zeros(batch, n_snippets, 1))
    return visual, audio, mask, reliability



def feeds_for(sess, inputs: tuple[torch.Tensor, ...]) -> dict:
    """Feeds for a session, keyed by the names the *graph* actually has.

    The tracer prunes unused arguments (early/late/cross fusion never read the reliability vector,
    so their graphs have three inputs), and hard-coding four feeds makes ONNX Runtime reject it.
    """
    provided = {"visual": inputs[0].numpy(), "audio": inputs[1].numpy(),
                "mask": inputs[2].numpy(), "reliability": inputs[3].numpy()}
    return {tensor.name: provided[tensor.name] for tensor in sess.get_inputs()}


def machine_load() -> dict:
    """Load average, so a latency number can be read in context (training may run in parallel)."""
    try:
        one, five, fifteen = open("/proc/loadavg").read().split()[:3]
        return {"loadavg_1m": float(one), "loadavg_5m": float(five),
                "loadavg_15m": float(fifteen)}
    except OSError:                                # non-Linux: no context, not a failure
        return {}


def onnx_latency(path: Path, inputs: tuple[torch.Tensor, ...], n_reliability: int,
                 runs: int, threads: int) -> dict:
    """p50/p95 wall-clock latency of one forward pass through an ONNX Runtime session."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])
    feeds = feeds_for(sess, inputs)
    for _ in range(3):                                    # warm-up
        sess.run(None, feeds)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - started) * 1e3)
    times.sort()
    p50 = statistics.median(times)
    return {
        "runs": runs,
        "p50_ms": round(p50, 3),
        "p95_ms": round(times[max(0, int(0.95 * len(times)) - 1)], 3),
        "snippets_per_second": round(inputs[0].shape[1] / (p50 / 1e3), 1),
    }


def onnx_outputs(path: Path, inputs: tuple[torch.Tensor, ...], n_reliability: int):
    """Snippet logits from an ONNX session (used for the fp32 <-> int8 parity check)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds_for(sess, inputs))[0]


def make_fp16(fp32_path: Path, fp16_path: Path) -> Path:
    """Internal weights -> float16, graph I/O stays float32 (Cast in/out at the boundary).

    The deployed runtime feeds float32 arrays, so the fp16 variant cannot simply retype the inputs:
    each graph input keeps its name and type, a ``Cast`` immediately converts it to fp16, and every
    internal consumer is rewritten to read the fp16 tensor. Each graph output gets a symmetric
    ``Cast`` back to fp32. All internal ops (MatMul, Gemm, Softmax, MaskedSoftmax, LayerNorm,
    ReLU/Sigmoid) accept fp16 at opset 17, and residual float constants are converted with the
    weights. If anything still breaks type uniformity, ONNX Runtime session creation fails loudly -
    the script's honesty rule is that a broken variant is reported, not shipped silently.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(fp32_path))
    graph = model.graph

    # 1. stored weights -> fp16
    for initial in list(graph.initializer):
        if initial.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(initial).astype(np.float16)
            replaced = numpy_helper.from_array(arr, name=initial.name)
            graph.initializer.remove(initial)
            graph.initializer.append(replaced)

    # 2. inline float constants -> fp16
    for node in graph.node:
        if node.op_type == "Constant":
            value = next(a for a in node.attribute if a.name == "value")
            if value.t.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(value.t).astype(np.float16)
                value.t.CopyFrom(numpy_helper.from_array(arr, name=value.t.name))

    # 2b. the traced MHA graph carries pre-existing ``Cast -> FLOAT`` nodes (no-ops in the fp32
    #     export) and ``ConstantOfShape`` fill values (the -inf of masked attention) that would
    #     keep fp32 inside the fp16 graph and break type unification. Convert both to fp16; my
    #     boundary casts (``cast_in_*``/``cast_out_*``) are exempt.
    for node in graph.node:
        if node.op_type == "ConstantOfShape":
            value = next(a for a in node.attribute if a.name == "value")
            if value.t.data_type == TensorProto.FLOAT:
                value.t.CopyFrom(numpy_helper.from_array(
                    numpy_helper.to_array(value.t).astype(np.float16), name=value.t.name))
        elif node.op_type != "Cast" or node.name.startswith(("cast_in_", "cast_out_")):
            continue
        else:
            to_attr = next(a for a in node.attribute if a.name == "to")
            if to_attr.i == TensorProto.FLOAT:
                to_attr.i = TensorProto.FLOAT16

    # 3. cast in: only float inputs (the bool `mask` keeps its type)
    float_inputs = {i.name for i in graph.input
                    if i.type.tensor_type.elem_type == TensorProto.FLOAT}
    cast_in = [helper.make_node("Cast", [name], [f"{name}__fp16"],
                                to=TensorProto.FLOAT16, name=f"cast_in_{name}")
               for name in float_inputs]
    for node in graph.node:
        renamed = [f"{a}__fp16" if a in float_inputs else a for a in node.input]
        del node.input[:]
        node.input.extend(renamed)

    # 4. cast out: the producer of each output is renamed to <out>__fp16, then Cast back
    output_names = [o.name for o in graph.output]
    for node in graph.node:
        renamed = [f"{a}__fp16" if a in output_names else a for a in node.output]
        del node.output[:]
        node.output.extend(renamed)
    cast_out = [helper.make_node("Cast", [f"{name}__fp16"], [name],
                                 to=TensorProto.FLOAT, name=f"cast_out_{name}")
                for name in output_names]

    # 5. intermediate type hints
    for value_info in graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

    for i, node in enumerate(cast_in):          # cast-in must precede its consumers
        graph.node.insert(i, node)
    graph.node.extend(cast_out)
    onnx.checker.check_model(model)
    onnx.save(model, str(fp16_path))
    return fp16_path



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True,
                        help="checkpoint produced by safewatch.train[_fusion]")
    parser.add_argument("--out-dir", default="export/onnx")
    parser.add_argument("--snippets", type=int, default=100, help="time steps in the dummy batch")
    parser.add_argument("--bench", type=int, default=30, help="latency repetitions")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--int8", action="store_true", help="also build the quantised model")
    parser.add_argument("--per-channel", action="store_true",
                        help="int8: quantise MatMul weights per output channel (the Session-9c "
                             "rejections used per-tensor; per-channel keeps one scale per channel "
                             "and typically halves the logit drift)")
    parser.add_argument("--fp16", action="store_true",
                        help="also build a float16 variant: fp16 weights + cast in/out, the "
                             "size-reduction option without int8's logit drift (mainly for ARM "
                             "targets; x86 latency is reported as measured, not assumed)")
    parser.add_argument("--parity-tolerance", type=float, default=0.05,
                        help="max |logit diff| accepted between fp32 and int8 snippet curves")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.ckpt).parent.name

    model, cfg, kind = load_model(args.ckpt, torch.device("cpu"))
    model.eval()
    stripped = strip_attention_weights(model)
    visual_dim, audio_dim, n_reliability = model_shape(model, cfg)
    inputs = make_inputs(visual_dim, audio_dim, n_reliability, args.snippets)
    names = ["visual", "audio", "mask"] + (["reliability"] if n_reliability else [])
    wrapper = ExportWrapper(model)

    print(f"[onnx] ckpt={args.ckpt} kind={kind} visual={visual_dim} audio={audio_dim} "
          f"reliability={n_reliability} snippets={args.snippets} attention_weights={not stripped}")

    # The PyTorch reference MUST be captured before exporting: the legacy TorchScript exporter
    # mutates the model in place (measured: the same model returns different logits after
    # `torch.onnx.export`, max |diff| ~8 on this corpus), so a "parity check" run afterwards sees a
    # changed model and reports a bogus mismatch (docs/journal.md, Session 9c).
    with torch.no_grad():
        torch_logits = wrapper(*inputs)[0].numpy().copy()

    fp32_path = out_dir / f"{stem}_fp32.onnx"
    dynamic_axes = {name: {0: "batch", 1: "time"} for name in names}
    # ``dynamo=False`` selects the TorchScript exporter: the dynamo path would pull in `onnxscript`,
    # a dependency this CPU-only repo deliberately does not carry (see requirements.txt), and the
    # graph here is a plain feed-forward head with a dynamic time axis.
    torch.onnx.export(wrapper, inputs, str(fp32_path), input_names=names,
                      output_names=["snippet_logits", "multi_logits"], dynamic_axes=dynamic_axes,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    report: dict = {
        "checkpoint": str(args.ckpt),
        "kind": kind,
        "device": f"cpu/{args.threads} threads",
        "shapes": {"visual_dim": visual_dim, "audio_dim": audio_dim,
                   "reliability_dim": n_reliability, "time_steps": args.snippets},
        "attention_weights_in_graph": False,
        "pytorch_reference_taken": "before export (the legacy exporter mutates the model in place)",
        "measurement_conditions": machine_load(),
        "fp32": {"file": fp32_path.name, "size_mb": round(fp32_path.stat().st_size / 2**20, 3)},
    }
    if kind == "fusion":
        report["fusion_mode"] = cfg.get("fusion_mode")

    report["fp32"]["max_abs_diff_vs_pytorch"] = float(
        np.abs(torch_logits - onnx_outputs(fp32_path, inputs, n_reliability)).max())
    report["fp32"].update(onnx_latency(fp32_path, inputs, n_reliability, args.bench, args.threads))

    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = out_dir / (f"{stem}_int8_pc.onnx" if args.per_channel
                               else f"{stem}_int8.onnx")
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                         per_channel=args.per_channel)
        max_diff = float(np.abs(torch_logits
                                - onnx_outputs(int8_path, inputs, n_reliability)).max())
        report["int8"] = {
            "file": int8_path.name,
            "per_channel": bool(args.per_channel),
            "size_mb": round(int8_path.stat().st_size / 2**20, 3),
            "max_abs_diff_vs_pytorch_logits": max_diff,
            "accepted": bool(max_diff <= args.parity_tolerance),
            "parity_tolerance": args.parity_tolerance,
        }
        report["int8"].update(onnx_latency(int8_path, inputs, n_reliability,
                                           args.bench, args.threads))
        report["size_ratio_int8_over_fp32"] = round(
            report["int8"]["size_mb"] / report["fp32"]["size_mb"], 3)
        report["speedup_int8"] = round(report["fp32"]["p50_ms"] / report["int8"]["p50_ms"], 3)
        if not report["int8"]["accepted"]:
            print(f"[onnx] WARNING int8{' (per-channel)' if args.per_channel else ''} shifts "
                  f"snippet logits by {max_diff:.4f} "
                  f"(> {args.parity_tolerance}) - do not deploy without recalibration")

    if args.fp16:
        fp16_path = out_dir / f"{stem}_fp16.onnx"
        make_fp16(fp32_path, fp16_path)
        diff = float(np.abs(torch_logits
                            - onnx_outputs(fp16_path, inputs, n_reliability)).max())
        report["fp16"] = {
            "file": fp16_path.name,
            "size_mb": round(fp16_path.stat().st_size / 2**20, 3),
            "max_abs_diff_vs_pytorch_logits": diff,
            "accepted": bool(diff <= args.parity_tolerance),
            "parity_tolerance": args.parity_tolerance,
        }
        report["fp16"].update(onnx_latency(fp16_path, inputs, n_reliability,
                                           args.bench, args.threads))
        report["size_ratio_fp16_over_fp32"] = round(
            report["fp16"]["size_mb"] / report["fp32"]["size_mb"], 3)
        report["speedup_fp16"] = round(report["fp32"]["p50_ms"] / report["fp16"]["p50_ms"], 3)

    report_path = out_dir / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[onnx] report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
