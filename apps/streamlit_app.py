"""Live agent graph + evaluation dashboard (Part 2 and Part 3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.graph_render import render_agent_graph  # noqa: E402
from agents.workflow import GRAPH_EDGES, GRAPH_NODES, run_agent_loop, save_loop_chart  # noqa: E402
from adaptive.compare import compare  # noqa: E402
from adaptive.eval_suite import run_suite  # noqa: E402
from adaptive.graph import graph_mermaid, run_adaptive  # noqa: E402
from pipeline.config import REPORTS_DIR  # noqa: E402

st.set_page_config(page_title="Week 11 Agent Workflows", layout="wide")
st.title("Week 11 — Agent-based ML workflows")
st.caption("Part 2 live graph (modified in-class loop) and Part 3 adaptive LangGraph.")

tab_loop, tab_adaptive, tab_eval = st.tabs(
    ["Part 2: Agent training loop", "Part 3: Adaptive graph", "Evaluation vs baselines"]
)


def _plotly_graph(active: str | None = None) -> go.Figure:
    coords = {
        "DataLoader": (0, 0.2),
        "FeatureEngineer": (1, 0.7),
        "Trainer": (2, 0.7),
        "Evaluator": (3, 0.7),
        "Tuner": (2, 0.05),
        "Reporter": (4, 0.2),
    }
    edge_x, edge_y = [], []
    for src, dst in GRAPH_EDGES:
        edge_x += [coords[src][0], coords[dst][0], None]
        edge_y += [coords[src][1], coords[dst][1], None]
    node_x = [coords[n["id"]][0] for n in GRAPH_NODES]
    node_y = [coords[n["id"]][1] for n in GRAPH_NODES]
    colors = ["#f4a261" if n["id"] == active else "#457b9d" for n in GRAPH_NODES]
    if active is None:
        colors = ["#2a9d8f" if n["id"] == "Reporter" else "#457b9d" for n in GRAPH_NODES]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1.5, color="#adb5bd"), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[n["id"] for n in GRAPH_NODES],
            textposition="bottom center",
            marker=dict(size=28, color=colors),
            hovertext=[n["role"] for n in GRAPH_NODES],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=340,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=10, b=40),
        showlegend=False,
        plot_bgcolor="white",
    )
    return fig


with tab_loop:
    st.subheader("Modified agents")
    st.markdown(
        """
The in-class template was a straight **Loader → Train → Evaluate** chain.
This version adds a **FeatureEngineer**, a **Tuner** that expands features when
accuracy is low, and a **Reporter**. The coordinator is still a `for` loop, and
the graph below updates as each agent fires.
"""
    )
    target = st.slider("Accuracy target", 0.70, 0.95, 0.88, 0.01)
    placeholder = st.empty()
    log_box = st.empty()
    placeholder.plotly_chart(_plotly_graph(), use_container_width=True, key="agent_graph_idle")

    if st.button("Run agent loop", type="primary"):
        logs = []

        def on_event(events, meta):
            last = events[-1]
            logs.append(f"{last.ts}  **{last.agent}** [{last.status}] {last.detail}")
            placeholder.plotly_chart(
                _plotly_graph(active=last.agent),
                use_container_width=True,
                key=f"agent_graph_{len(events)}_{last.agent}",
            )
            log_box.markdown("\n\n".join(logs[-8:]))

        summary = run_agent_loop(target_accuracy=target, on_event=on_event)
        save_loop_chart(summary)
        render_agent_graph()
        placeholder.plotly_chart(
            _plotly_graph(active="Reporter"),
            use_container_width=True,
            key="agent_graph_done",
        )
        st.success(
            f"Final accuracy {summary['final_accuracy']:.3f}  |  F1 {summary['final_f1']:.3f}  |  "
            f"rounds {summary['rounds']}  |  met target={summary['met_target']}"
        )
        st.dataframe(pd.DataFrame(summary["history"]), use_container_width=True)
        st.json({k: v for k, v in summary.items() if k not in {"events", "history"}})

with tab_adaptive:
    st.subheader("LangGraph adaptive workflow")
    st.markdown(
        "The graph **selects the next agent from intermediate metrics**, retries failed "
        "nodes, and stops when accuracy crosses the target."
    )
    st.code(graph_mermaid(), language="mermaid")
    if st.button("Run adaptive graph", type="primary"):
        result = run_adaptive(target_accuracy=0.82, max_rounds=5)
        st.metric("Accuracy", f"{result.get('metrics', {}).get('accuracy', 0):.3f}")
        st.metric("Rounds", result.get("round"))
        st.write("Route:", result.get("route"), " | met target:", result.get("met_target"))
        st.dataframe(pd.DataFrame(result.get("history") or []), use_container_width=True)

with tab_eval:
    st.subheader("Evaluation suite and baseline comparison")
    if st.button("Run evaluation suite"):
        results = run_suite()
        df = pd.DataFrame(
            [
                {
                    "case": r.case_id,
                    "passed": r.passed,
                    "score": r.score,
                    "threshold": r.threshold,
                    "expected": r.expected,
                    "observed": r.observed,
                }
                for r in results
            ]
        )
        st.dataframe(df, use_container_width=True)
        st.write(f"Pass rate: {df['passed'].mean():.0%}")
        (REPORTS_DIR / "eval_suite.json").write_text(
            json.dumps(
                [
                    {
                        "case_id": r.case_id,
                        "passed": r.passed,
                        "score": r.score,
                        "threshold": r.threshold,
                        "expected": r.expected,
                        "observed": r.observed,
                        **r.details,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    if st.button("Compare against class baselines"):
        cmp = compare()
        st.dataframe(cmp, use_container_width=True)
        st.image(str(REPORTS_DIR / "baseline_comparison.png"))
