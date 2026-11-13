"""Read cleaned data from the database, visualize it, and train a baseline model."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from pipeline.config import (
    HIGH_DEMAND_THRESHOLD_QUANTILE,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_dirs,
)
from pipeline.load import read_merged

NUMERIC = ["temp_c", "humidity", "wind_kph", "precip_mm", "is_weekend", "month", "day_of_week"]
CATEGORICAL = ["station_id", "weather_code"]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cutoff = out["rentals"].quantile(HIGH_DEMAND_THRESHOLD_QUANTILE)
    out["high_demand"] = (out["rentals"] > cutoff).astype(int)
    return out


def train_baseline(df: pd.DataFrame) -> dict:
    labeled = add_target(df)
    x = labeled[NUMERIC + CATEGORICAL]
    y = labeled["high_demand"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=400, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    metrics = {
        "model": "logistic_regression",
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "positive_rate": float(y.mean()),
        "report": classification_report(y_test, preds, output_dict=True),
    }
    ensure_dirs()
    joblib.dump(model, MODELS_DIR / "baseline_logreg.pkl")
    (REPORTS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted high demand")
    ax.set_ylabel("Actual high demand")
    ax.set_title(f"Baseline accuracy {metrics['accuracy']:.3f}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "baseline_confusion_matrix.png", dpi=140)
    plt.close(fig)
    return metrics


def make_visualizations(df: pd.DataFrame) -> list[str]:
    ensure_dirs()
    paths = []
    daily = df.groupby("ride_date", as_index=False)["rentals"].sum()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(daily["ride_date"], daily["rentals"], color="#1f4e79", linewidth=1)
    ax.set_title("Citywide daily bike rentals")
    ax.set_ylabel("Rentals")
    fig.tight_layout()
    p1 = REPORTS_DIR / "daily_rentals.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    paths.append(str(p1))

    fig, ax = plt.subplots(figsize=(6, 4))
    sample = df.sample(min(len(df), 2500), random_state=RANDOM_STATE)
    sns.scatterplot(
        data=sample,
        x="temp_c",
        y="rentals",
        hue="precip_mm",
        palette="coolwarm",
        s=18,
        ax=ax,
        legend=False,
    )
    ax.set_title("Rentals vs temperature (color = precipitation)")
    fig.tight_layout()
    p2 = REPORTS_DIR / "rentals_vs_temp.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    paths.append(str(p2))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.barplot(data=df, x="station_name", y="rentals", estimator="mean", errorbar=None, ax=ax)
    ax.set_title("Average daily rentals by station")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    p3 = REPORTS_DIR / "rentals_by_station.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    paths.append(str(p3))
    return paths


def run_analysis(uri: str | None = None) -> dict:
    df = read_merged(uri)
    charts = make_visualizations(df)
    metrics = train_baseline(df)
    metrics["charts"] = charts
    metrics["n_rows"] = int(len(df))
    return metrics
