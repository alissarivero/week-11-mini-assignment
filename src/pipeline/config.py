"""Shared paths and database settings for the bike-demand pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow"))
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

BIKE_RAW = DATA_RAW / "bike_rentals.csv"
WEATHER_RAW = DATA_RAW / "weather.csv"

BIKE_CLEAN = DATA_PROCESSED / "bike_rentals_clean.csv"
WEATHER_CLEAN = DATA_PROCESSED / "weather_clean.csv"
MERGED_CSV = DATA_PROCESSED / "bike_weather_merged.csv"

TABLE_NAME = "bike_weather_daily"
HIGH_DEMAND_THRESHOLD_QUANTILE = 0.50
RANDOM_STATE = 42

ANALYTICS_DB_URI = os.getenv(
    "ANALYTICS_DB_URI",
    f"sqlite:///{PROJECT_ROOT / 'data' / 'analytics.db'}",
)


def ensure_dirs() -> None:
    for path in (DATA_RAW, DATA_PROCESSED, REPORTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
