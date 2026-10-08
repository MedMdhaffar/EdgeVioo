"""P6 - the supervision console is driven headlessly with Streamlit's own ``AppTest``.

This is the only test that exercises `app.py`, and it earns its runtime: the console is where the
operator's verdict enters the project, and the wiring (sidebar -> clip queue -> timeline ->
decision log) cannot be covered by unit-testing the state layer alone. It runs against a
**fabricated** run directory, so it is hermetic and fast, and it asserts the two things that matter:

* the app renders without exceptions on a minimal run, and says so when no incident sheet exists
  (it must never fake an alert);
* clicking "Confirmer" appends one line with clip, score and segments.

Run:  .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


def _fake_run(root: Path) -> Path:
    """A two-clip run: one incident (20 snippets above threshold) and one quiet clip."""
    run = root / "runs" / "2026-09-30_demo_cross"
    run.mkdir(parents=True)
    incident = np.concatenate([np.zeros(30), np.full(20, 0.9), np.zeros(30)]).astype(np.float32)
    np.savez_compressed(run / "predictions.npz",
                        **{"scores::clip_a": incident,
                           "gt::clip_a": np.zeros(80, dtype=np.int64),
                           "scores::clip_b": np.zeros(80, dtype=np.float32),
                           "gt::clip_b": np.zeros(80, dtype=np.int64)})
    (run / "config.json").write_text(json.dumps({"fusion_mode": "cross"}), encoding="utf-8")
    (run / "test_metrics.json").write_text(json.dumps(
        {"stride_frames": 16.0, "n_clips": 2, "macro_per_class_ap": 0.5,
         "ap": {"global__pr_auc": 0.7, "per_video__average_precision": 0.8}}), encoding="utf-8")
    return root / "runs"


def _set_text(app, label: str, value: str) -> None:
    """Set a text input **by label**: AppTest orders elements main-pane-first, so index-based access
    silently targets the wrong widget (it once set the clip search box to a directory path)."""
    for widget in app.text_input:
        if widget.label == label:
            widget.set_value(value)
            return
    raise AssertionError(f"no text input labelled {label!r}")


def _launch(tmp_path: Path, decisions: Path):
    runs_dir = _fake_run(tmp_path)
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    _set_text(app, "Runs Directory", str(runs_dir))
    _set_text(app, "Incident Sheets Directory", str(tmp_path / "sheets"))   # empty on purpose
    _set_text(app, "Decision Audit Log", str(decisions))
    app.run()
    return app


def test_console_renders_without_sheets_and_never_fakes_an_alert(tmp_path: Path):
    app = _launch(tmp_path, tmp_path / "decisions.jsonl")

    assert len(app.exception) == 0, [getattr(e, "value", e) for e in app.exception]
    subheaders = [block.value for block in app.subheader]
    assert "Incident Sheet" in subheaders and "Operator Decision" in subheaders
    # no sheet on disk -> a warning plus the command that would produce one, never a filled card
    assert any("No incident sheet" in warning.value for warning in app.warning)
    assert any("incident_sheet.py" in block.value for block in app.code)
    buttons = [button.label for button in app.button][:3]
    assert buttons == ["✅ Confirm", "⛔ Reject", "❔ Unsure"]


def test_operator_decision_is_appended_with_its_context(tmp_path: Path):
    decisions = tmp_path / "decisions.jsonl"
    app = _launch(tmp_path, decisions)

    app.button[0].click().run()                             # "Confirmer" on the first queue item
    assert len(app.exception) == 0, [getattr(e, "value", e) for e in app.exception]

    lines = [line for line in decisions.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["decision"] == "confirmed"
    assert record["clip_id"] == "clip_a"                     # highest score -> first in the queue
    assert record["clip_score"] == pytest.approx(0.9, abs=1e-6)
    assert record["n_segments"] == 1                         # 20 snippets * 16/24 s > 5.33 s rule
    assert record["run"] == "2026-09-30_demo_cross"

    # a second verdict on the same clip appends rather than overwriting (append-only audit trail)
    app.button[1].click().run()                              # "Rejeter"
    lines = [line for line in decisions.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["decision"] == "confirmed"
    assert json.loads(lines[1])["decision"] == "rejected"


def test_live_demo_warns_when_the_checkpoint_cannot_run_live(tmp_path: Path):
    """A VideoSwin/VGGish champion must raise a visible warning, not silently show ~0 scores.

    The upload path can only reproduce I3D visual + log-mel audio; a checkpoint trained on another
    feature space scores near zero on obvious incidents (measured), so the console has to say so.
    """
    import torch

    from safewatch.models.fusion import FusionModel

    runs_dir = _fake_run(tmp_path)
    run = runs_dir / "2026-09-30_swin_demo"
    run.mkdir()
    model = FusionModel(visual_dim=768, audio_dim=132, emb=32, hidden=64, n_categories=6,
                        fusion_mode="late")
    torch.save({"model": model.state_dict(),
                "config": {"fusion_mode": "late", "feature_set": "swin_rgb",
                           "audio_dir": "data/features/mel"}}, str(run / "ckpt_best.pt"))

    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    _set_text(app, "Runs Directory", str(runs_dir))
    _set_text(app, "Incident Sheets Directory", str(tmp_path / "sheets"))
    _set_text(app, "Decision Audit Log", str(tmp_path / "decisions.jsonl"))
    for radio in app.radio:
        if radio.label == "Mode":
            radio.set_value("Live Demo / Import Video")
    app.run()

    assert len(app.exception) == 0, [getattr(e, "value", e) for e in app.exception]
    assert any("different feature space" in warning.value for warning in app.warning)

