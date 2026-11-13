"""Agents that load cleaned bike/weather data, train, evaluate, and tune."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agents.base import Agent
from pipeline.analyze import add_target
from pipeline.config import RANDOM_STATE
from pipeline.load import read_merged

FEATURE_STAGES = [
    ["humidity"],
    ["humidity", "wind_kph"],
    ["temp_c", "precip_mm", "is_weekend"],
    [
        "temp_c",
        "humidity",
        "wind_kph",
        "precip_mm",
        "is_weekend",
        "month",
        "day_of_week",
        "station_id",
        "weather_code",
    ],
]


def load_data(uri: str | None = None) -> pd.DataFrame:
    df = read_merged(uri)
    return add_target(df)


def select_features(df: pd.DataFrame, stage: int) -> dict[str, Any]:
    cols = FEATURE_STAGES[min(stage, len(FEATURE_STAGES) - 1)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return {"stage": stage, "columns": cols, "n_rows": int(len(df))}


def train_model(df: pd.DataFrame, columns: list[str], model_name: str = "logreg") -> dict[str, Any]:
    numeric = [c for c in columns if c not in {"station_id", "weather_code"}]
    categorical = [c for c in columns if c in {"station_id", "weather_code"}]
    x = df[columns]
    y = df["high_demand"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    clf = (
        RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE)
        if model_name == "random_forest"
        else LogisticRegression(max_iter=400, random_state=RANDOM_STATE)
    )
    pipe = Pipeline([("prep", ColumnTransformer(transformers)), ("clf", clf)])
    pipe.fit(x_train, y_train)
    return {
        "model": pipe,
        "model_name": model_name,
        "x_test": x_test,
        "y_test": y_test,
        "n_features": len(columns),
    }


def evaluate_model(bundle: dict[str, Any]) -> dict[str, Any]:
    preds = bundle["model"].predict(bundle["x_test"])
    acc = float(accuracy_score(bundle["y_test"], preds))
    f1 = float(f1_score(bundle["y_test"], preds))
    return {
        "accuracy": acc,
        "f1": f1,
        "model_name": bundle["model_name"],
        "n_features": bundle["n_features"],
        "n_test": int(len(bundle["y_test"])),
    }


def tune(stage: int, metrics: dict[str, Any], target: float) -> dict[str, Any]:
    """Decide the next experiment: add features, then switch to a forest."""
    if metrics["accuracy"] >= target:
        return {"action": "stop", "stage": stage, "model_name": metrics["model_name"]}
    if stage < len(FEATURE_STAGES) - 1:
        return {
            "action": "add_features",
            "stage": stage + 1,
            "model_name": "logreg",
            "reason": f"accuracy {metrics['accuracy']:.3f} < {target}",
        }
    return {
        "action": "switch_model",
        "stage": stage,
        "model_name": "random_forest",
        "reason": "feature set exhausted; try a non-linear model",
    }


def build_agents() -> dict[str, Agent]:
    return {
        "loader": Agent("DataLoader", "read cleaned table from the database", load_data),
        "features": Agent("FeatureEngineer", "choose the active feature set", select_features),
        "trainer": Agent("Trainer", "fit a classifier on the current features", train_model),
        "evaluator": Agent("Evaluator", "score hold-out accuracy and F1", evaluate_model),
        "tuner": Agent("Tuner", "expand features or change the estimator", tune),
        "reporter": Agent("Reporter", "write the final loop summary", lambda **kw: kw),
    }
