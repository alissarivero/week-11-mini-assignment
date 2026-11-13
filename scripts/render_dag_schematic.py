"""Static DAG graph for the write-up (mirrors the Airflow Graph view)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.config import REPORTS_DIR, ensure_dirs  # noqa: E402


def main() -> None:
    ensure_dirs()
    g = nx.DiGraph()
    nodes = [
        "ingest.ingest_bikes",
        "ingest.ingest_weather",
        "clean_bikes.strip_and_validate",
        "clean_weather.interpolate_and_clip",
        "merge",
        "load",
        "analyze",
    ]
    g.add_nodes_from(nodes)
    g.add_edges_from(
        [
            ("ingest.ingest_bikes", "clean_bikes.strip_and_validate"),
            ("ingest.ingest_weather", "clean_weather.interpolate_and_clip"),
            ("clean_bikes.strip_and_validate", "merge"),
            ("clean_weather.interpolate_and_clip", "merge"),
            ("merge", "load"),
            ("load", "analyze"),
        ]
    )
    pos = {
        "ingest.ingest_bikes": (0, 1),
        "ingest.ingest_weather": (0, 0),
        "clean_bikes.strip_and_validate": (1.4, 1),
        "clean_weather.interpolate_and_clip": (1.4, 0),
        "merge": (2.8, 0.5),
        "load": (4.0, 0.5),
        "analyze": (5.2, 0.5),
    }
    fig, ax = plt.subplots(figsize=(12, 4))
    nx.draw(
        g,
        pos,
        with_labels=True,
        node_color="#1d3557",
        font_color="white",
        font_size=7,
        node_size=2800,
        arrows=True,
        ax=ax,
    )
    ax.set_title("bike_demand_etl DAG — parallel ingest/clean, then merge → load → analyze")
    fig.tight_layout()
    dest = REPORTS_DIR / "dag_graph.png"
    fig.savefig(dest, dpi=150, facecolor="white")
    plt.close(fig)
    print(dest)


if __name__ == "__main__":
    main()
