"""P6 - plotly figures for the supervision console (pure: no Streamlit, no Streamlit state).

Kept apart from ``app.py`` for the same reason ``explain/report.py`` is apart from
``scripts/incident_sheet.py``: a figure that can be built from a plain dict can be unit-tested and
reused by a future FastAPI/mobile front-end, and the app stays a router of widgets.

All x-axes are in **seconds** - the timeline dict it consumes has already converted the snippet grid
(:func:`safewatch.ui.state.timeline`), so a mirror-grid run and an official-grid run plot on the
same scale instead of looking 4x different.
"""
from __future__ import annotations

import plotly.graph_objects as go

PROBABILITY_COLOUR = "#1f77b4"
SEGMENT_COLOUR = "rgba(255, 127, 14, 0.25)"
GROUND_TRUTH_COLOUR = "rgba(44, 160, 44, 0.18)"


def timeline_figure(timeline: dict, title: str | None = None, height: int = 330) -> go.Figure:
    """Score curve + alert threshold + predicted segments + (optional) annotated intervals.

    Ground truth is drawn but never used to compute anything: it is the evaluation reference, and
    showing it for a clip of the **test** split is explicitly labelled in the app so an operator
    cannot mistake it for a model output.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=timeline["seconds"], y=timeline["probability"], mode="lines", name="score",
        line={"color": PROBABILITY_COLOUR, "width": 2},
        hovertemplate="%{x:.1f} s<br>p=%{y:.3f}<extra></extra>"))
    figure.add_hline(y=timeline["threshold"], line={"dash": "dot", "color": "#d62728"},
                     annotation_text=f"threshold {timeline['threshold']:g}",
                     annotation_position="top left")

    for index, segment in enumerate(timeline["segments"], start=1):
        figure.add_vrect(x0=segment["start_s"], x1=segment["end_s"], fillcolor=SEGMENT_COLOUR,
                         line_width=0, layer="below",
                         annotation_text=f"A{index}", annotation_position="top left")
    for interval in timeline.get("gt_intervals") or []:
        figure.add_vrect(x0=interval["start_s"], x1=interval["end_s"],
                         fillcolor=GROUND_TRUTH_COLOUR, line_width=0, layer="below")

    figure.update_layout(
        title=title or timeline["clip_id"],
        height=height, margin={"l": 40, "r": 20, "t": 50, "b": 40},
        xaxis_title="Time (s)", yaxis_title="Incident Probability",
        yaxis={"range": [0, 1.02]}, showlegend=False,
        hovermode="x unified")
    return figure


def category_figure(category: dict, height: int = 260) -> go.Figure | None:
    """Horizontal bars of the ranked category probabilities, or ``None`` when unavailable."""
    ranked = (category or {}).get("ranked") or []
    if not ranked:
        return None
    labels = [row["category"] for row in reversed(ranked)]
    values = [row["probability"] for row in reversed(ranked)]
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                              marker={"color": PROBABILITY_COLOUR}))
    figure.update_layout(height=height, margin={"l": 10, "r": 20, "t": 30, "b": 30},
                         xaxis={"range": [0, 1], "title": "Probability (uncalibrated head)"},
                         yaxis_title=None, showlegend=False)
    return figure
