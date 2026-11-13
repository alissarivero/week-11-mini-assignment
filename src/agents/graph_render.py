"""Render the live agent graph to PNG so the submission includes a static view."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from agents.workflow import GRAPH_EDGES, GRAPH_NODES
from pipeline.config import REPORTS_DIR, ensure_dirs


def render_agent_graph(active: str | None = None, dest: Path | None = None) -> Path:
    ensure_dirs()
    dest = dest or (REPORTS_DIR / "agent_graph.png")
    graph = nx.DiGraph()
    for node in GRAPH_NODES:
        graph.add_node(node["id"], role=node["role"])
    graph.add_edges_from(GRAPH_EDGES)

    pos = {
        "DataLoader": (0, 0.2),
        "FeatureEngineer": (1, 0.6),
        "Trainer": (2, 0.6),
        "Evaluator": (3, 0.6),
        "Tuner": (2, 0.0),
        "Reporter": (4, 0.2),
    }
    colors = []
    for node in graph.nodes:
        if node == active:
            colors.append("#f4a261")
        elif node == "Tuner":
            colors.append("#e9c46a")
        elif node == "Reporter":
            colors.append("#2a9d8f")
        else:
            colors.append("#457b9d")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=2200, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white", ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        edge_color="#6c757d",
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    ax.set_title("Modified agent graph: feature expansion loop + reporter")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest
