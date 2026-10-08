"""Regenerate every figure in ``docs/report/figures/`` from audited artefacts.

Sources (nothing is hardcoded that already lives on disk):
  fig1 ladder      <- runs/*/test_metrics.json (AP global + per-clip)
  fig2 per-class   <- runs/2026-09-29_p3vggish30ep_cross/test_metrics.json
  fig3 seeds       <- data/logs/seed_margin_table.txt (paired margin block)
  fig4 robustness  <- data/logs/robustness_table.txt (cross sweeps only)
  fig5 calibration <- runs/2026-09-29_p3vggish30ep_cross/calibration.json

Usage:
    .venv/bin/python scripts/make_report_figures.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LADDER = [
    ("p2vggish30ep_visual", "visual seul"),
    ("p2vggish30ep_audio", "audio seul"),
    ("p3vggish30ep_late", "late"),
    ("p3vggish30ep_cross", "cross (ref)"),
    ("p3vggish30ep_early", "early"),
    ("p3vggish30ep_adaptive", "adaptive"),
]
CLASSES = ["fighting", "shooting", "riot", "abuse", "car_accident", "explosion"]
REF_RUN = ROOT / "runs" / "2026-09-29_p3vggish30ep_cross"


def latest_run(tag: str) -> Path:
    runs = sorted(ROOT.glob(f"runs/*_{tag}"))
    assert runs, f"no run for {tag}"
    return runs[-1]


def test_metrics(tag: str) -> dict:
    return json.loads((latest_run(tag) / "test_metrics.json").read_text(encoding="utf-8"))


def fig1_ladder() -> None:
    labels, glob, clip = [], [], []
    for tag, label in LADDER:
        m = test_metrics(tag)["ap"]
        labels.append(label)
        glob.append(m["global__pr_auc"])
        clip.append(m["per_video__average_precision"])
    x = range(len(labels))
    w = 0.38
    _, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar([i - w / 2 for i in x], glob, w, label="AP globale")
    ax.bar([i + w / 2 for i in x], clip, w, label="AP per-clip")
    ax.set_xticks(list(x), labels, rotation=12)
    ax.set_ylim(0.55, 0.85)
    ax.set_ylabel("AP")
    ax.set_title("Officiel 30 epochs : les fusions battent le visuel seul")
    ax.legend()
    for g, c, i in zip(glob, clip, x, strict=True):
        ax.text(i - w / 2, g + 0.006, f"{g:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, c + 0.006, f"{c:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig1_ladder.png", dpi=120)
    plt.close()


def fig2_perclass() -> None:
    m = json.loads((REF_RUN / "test_metrics.json").read_text(encoding="utf-8"))
    pc = m["per_class_clip_level_ap"]
    vals = [pc[c]["ap"] for c in CLASSES]
    _, ax = plt.subplots(figsize=(9.5, 4.75))
    bars = ax.bar(CLASSES, vals)
    bars[CLASSES.index("abuse")].set_color("tab:red")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AP par classe (clip-level)")
    ax.set_title("Reference cross : 3 classes > 0.85, shooting 0.56, abuse 0.08")
    for v, c in zip(vals, CLASSES, strict=True):
        ax.text(c, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_perclass.png", dpi=120)
    plt.close()


def fig3_seeds() -> None:
    txt = (ROOT / "data" / "logs" / "seed_margin_table.txt").read_text(encoding="utf-8")

    block = txt.split("paired margin")[1]
    rows = re.findall(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)", block, re.M)
    assert len(rows) >= 3, "paired margin block missing"
    seeds = [r[0] for r in rows]
    margins = [float(r[3]) for r in rows]
    _, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(seeds, margins)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("marge cross - visuel (AP globale)")
    ax.set_title("Marge appariee par seed : signe identique, +0.040 a +0.049")
    for s, mg in zip(seeds, margins, strict=True):
        ax.text(s, mg + 0.001, f"{mg:+.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig3_seeds.png", dpi=120)
    plt.close()


def fig4_robustness() -> None:
    rows = (ROOT / "data" / "logs" / "robustness_table.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    att = [
        (float(r.split()[3]), float(r.split()[4]))
        for r in rows
        if "2026-09-29_p3vggish30ep_cross attenuation" in r
    ]
    occ = [
        (float(r.split()[3]), float(r.split()[4]))
        for r in rows
        if "2026-09-29_p3vggish30ep_cross occlusion" in r
    ]
    noi = [
        (float(r.split()[3]), float(r.split()[4]))
        for r in rows
        if "2026-09-29_p3vggish30ep_cross noise" in r
    ]
    _, ax = plt.subplots(figsize=(8.75, 4.5))
    ax.plot([a for a, _ in att], [v for _, v in att], "o-", label="attenuation audio")
    ax.plot([o for o, _ in occ], [v for _, v in occ], "s-", label="occlusion visuelle")
    ax.set_xlabel("niveau (gain audio / secondes video)")
    ax.set_ylabel("AP globale")
    ax.set_title("Robustesse : degradation progressive, pas d'effondrement")
    ax.legend()
    ax2 = ax.twiny()
    ax2.plot([n for n, _ in noi], [v for _, v in noi], "^-", color="gray",
             label="bruit audio (dB SNR)")
    ax2.set_xlabel("bruit audio (dB SNR)")
    ax2.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIG / "fig4_robustness.png", dpi=120)
    plt.close()


def fig5_calibration() -> None:
    c = json.loads((REF_RUN / "calibration.json").read_text(encoding="utf-8"))
    rv = c["reliability_val"]
    edges = rv["bin_edges"]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(rv["confidence"]))]
    _, ax = plt.subplots(figsize=(6.25, 4.75))
    ax.plot([0, 1], [0, 1], "k--", label="parfaitement calibre")
    ax.plot(centres, rv["accuracy"], "o-", label="exactitude observee (val, n=150)")
    ax.set_xlabel("confiance moyenne du bin")
    ax.set_ylabel("exactitude du bin")
    ax.set_title(f"Calibration : T={c['temperature']:.2f}, NLL "
                 f"{c['metrics']['nll_before']:.3f} -> {c['metrics']['nll_after']:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig5_calibration.png", dpi=120)
    plt.close()


def main() -> None:
    fig1_ladder()
    fig2_perclass()
    fig3_seeds()
    fig4_robustness()
    fig5_calibration()
    print("wrote", sorted(p.name for p in FIG.glob("fig*.png")))


if __name__ == "__main__":
    main()
