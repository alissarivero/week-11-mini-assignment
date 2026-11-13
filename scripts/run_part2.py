"""Run the Part 2 agent loop and export graph + metric screenshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.graph_render import render_agent_graph  # noqa: E402
from agents.workflow import run_agent_loop, save_loop_chart  # noqa: E402


def main() -> None:
    summary = run_agent_loop()
    chart = save_loop_chart(summary)
    graph = render_agent_graph()
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))
    print("accuracy chart:", chart)
    print("agent graph:", graph)


if __name__ == "__main__":
    main()
