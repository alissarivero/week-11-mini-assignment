"""Compare the adaptive graph against the two class baselines."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from adaptive.graph import run_adaptive
from agents.ml_agents import evaluate_model, load_data, select_features, train_model
from pipeline.analyze import run_analysis
from pipeline.config import REPORTS_DIR, ensure_dirs


def airflow_baseline() -> dict:
    metrics = run_analysis()
    return {
        "system": "fixed_airflow_dag",
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "steps": 8,
        "adaptive": False,
        "notes": "Predefined ingest→clean→merge→load→single logreg",
    }


def simple_agent_baseline() -> dict:
    df = load_data()
    feat = select_features(df, 0)
    bundle = train_model(df, feat["columns"], "logreg")
    metrics = evaluate_model(bundle)
    return {
        "system": "simple_in_class_agents",
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "steps": 3,
        "adaptive": False,
        "notes": "Loader → Trainer → Evaluator, one feature (humidity), no retry",
    }


def adaptive_system() -> dict:
    out = run_adaptive(target_accuracy=0.82, max_rounds=5)
    metrics = out.get("metrics") or {}
    return {
        "system": "adaptive_langgraph",
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "steps": 4 + 3 * int(out.get("round", 1)),
        "adaptive": True,
        "rounds": out.get("round"),
        "met_target": bool(metrics.get("accuracy", 0) >= 0.82),
        "notes": "Routes, retries, and expands features from intermediate accuracy",
    }


def compare() -> pd.DataFrame:
    rows = [airflow_baseline(), simple_agent_baseline(), adaptive_system()]
    df = pd.DataFrame(rows)
    ensure_dirs()
    df.to_csv(REPORTS_DIR / "baseline_comparison.csv", index=False)
    (REPORTS_DIR / "baseline_comparison.json").write_text(df.to_json(orient="records", indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(df["system"], df["accuracy"], color=["#1d3557", "#457b9d", "#2a9d8f"])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Hold-out accuracy")
    ax.set_title("Adaptive graph vs class baselines")
    ax.tick_params(axis="x", rotation=15)
    for i, val in enumerate(df["accuracy"]):
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "baseline_comparison.png", dpi=140)
    plt.close(fig)
    return df
