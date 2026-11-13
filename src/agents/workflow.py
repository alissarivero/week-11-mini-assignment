"""Coordinator loop: train/evaluate until the accuracy target is met (in-class for-loop)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agents.base import AgentEvent
from agents.ml_agents import FEATURE_STAGES, build_agents
from pipeline.config import REPORTS_DIR, ensure_dirs

DEFAULT_TARGET = 0.88
MAX_ROUNDS = 5

# Graph used by the live UI. Edges show the happy path plus the tune-back loop.
GRAPH_NODES = [
    {"id": "DataLoader", "role": "load cleaned data"},
    {"id": "FeatureEngineer", "role": "select features"},
    {"id": "Trainer", "role": "fit model"},
    {"id": "Evaluator", "role": "score accuracy"},
    {"id": "Tuner", "role": "adapt next round"},
    {"id": "Reporter", "role": "publish metrics"},
]
GRAPH_EDGES = [
    ("DataLoader", "FeatureEngineer"),
    ("FeatureEngineer", "Trainer"),
    ("Trainer", "Evaluator"),
    ("Evaluator", "Tuner"),
    ("Tuner", "FeatureEngineer"),
    ("Tuner", "Reporter"),
    ("Evaluator", "Reporter"),
]


def run_agent_loop(
    uri: str | None = None,
    target_accuracy: float = DEFAULT_TARGET,
    max_rounds: int = MAX_ROUNDS,
    on_event: Callable[[list[AgentEvent], dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    agents = build_agents()
    events: list[AgentEvent] = []
    history: list[dict[str, Any]] = []

    df = agents["loader"].run(events, uri=uri)
    if on_event:
        on_event(events, {"round": 0, "phase": "loaded", "rows": len(df)})

    stage = 0
    model_name = "logreg"
    final_metrics: dict[str, Any] = {}

    for round_idx in range(1, max_rounds + 1):
        feat = agents["features"].run(events, df=df, stage=stage)
        bundle = agents["trainer"].run(
            events, df=df, columns=feat["columns"], model_name=model_name
        )
        metrics = agents["evaluator"].run(events, bundle=bundle)
        decision = agents["tuner"].run(events, stage=stage, metrics=metrics, target=target_accuracy)
        row = {
            "round": round_idx,
            "stage": stage,
            "features": feat["columns"],
            "model_name": model_name,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "decision": decision["action"],
        }
        history.append(row)
        final_metrics = metrics
        if on_event:
            on_event(events, {"round": round_idx, "metrics": metrics, "decision": decision})

        if decision["action"] == "stop" or metrics["accuracy"] >= target_accuracy:
            break
        if model_name == "random_forest" and stage >= len(FEATURE_STAGES) - 1:
            break
        stage = decision["stage"]
        model_name = decision["model_name"]

    summary = {
        "target_accuracy": target_accuracy,
        "rounds": len(history),
        "best_accuracy": max(r["accuracy"] for r in history),
        "best_f1": max(r["f1"] for r in history),
        "final_accuracy": final_metrics.get("accuracy"),
        "final_f1": final_metrics.get("f1"),
        "final_model": final_metrics.get("model_name"),
        "history": history,
        "met_target": bool(final_metrics.get("accuracy", 0) >= target_accuracy),
        "n_rows": int(len(df)),
    }
    agents["reporter"].run(events, **{k: v for k, v in summary.items() if k != "history"})
    ensure_dirs()
    (REPORTS_DIR / "agent_loop_metrics.json").write_text(json.dumps(summary, indent=2))
    summary["events"] = [e.__dict__ for e in events]
    return summary


def save_loop_chart(summary: dict[str, Any], dest: Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    ensure_dirs()
    dest = dest or (REPORTS_DIR / "agent_loop_accuracy.png")
    rounds = [r["round"] for r in summary["history"]]
    acc = [r["accuracy"] for r in summary["history"]]
    f1 = [r["f1"] for r in summary["history"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(rounds, acc, marker="o", label="accuracy")
    ax.plot(rounds, f1, marker="s", label="F1")
    ax.axhline(summary["target_accuracy"], color="gray", linestyle="--", label="target")
    ax.set_xlabel("Loop round")
    ax.set_ylabel("Score")
    ax.set_title("Agent training loop — hold-out metrics")
    ax.set_xticks(rounds)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest
