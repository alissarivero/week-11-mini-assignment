"""Copy raw extracts into the working area so downstream tasks have a stable path."""

from __future__ import annotations

import shutil
from pathlib import Path

from pipeline.config import BIKE_RAW, DATA_PROCESSED, WEATHER_RAW, ensure_dirs


def ingest_bikes(src: Path | None = None) -> str:
    ensure_dirs()
    source = Path(src) if src else BIKE_RAW
    dest = DATA_PROCESSED / "bike_rentals_raw.csv"
    shutil.copy(source, dest)
    return str(dest)


def ingest_weather(src: Path | None = None) -> str:
    ensure_dirs()
    source = Path(src) if src else WEATHER_RAW
    dest = DATA_PROCESSED / "weather_raw.csv"
    shutil.copy(source, dest)
    return str(dest)
