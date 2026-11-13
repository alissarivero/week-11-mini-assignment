"""Run the Part 1 pipeline locally (same functions the DAG calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.analyze import run_analysis  # noqa: E402
from pipeline.config import BIKE_RAW, WEATHER_RAW  # noqa: E402
from pipeline.ingest import ingest_bikes, ingest_weather  # noqa: E402
from pipeline.load import load_merged  # noqa: E402
from pipeline.transform import clean_bikes, clean_weather, merge_datasets  # noqa: E402


def main() -> None:
    bike_raw = ingest_bikes(BIKE_RAW)
    weather_raw = ingest_weather(WEATHER_RAW)
    bike_clean = clean_bikes(bike_raw)
    weather_clean = clean_weather(weather_raw)
    merged = merge_datasets(bike_clean, weather_clean)
    loaded = load_merged(merged)
    metrics = run_analysis()
    print(json.dumps({"load": loaded, "accuracy": metrics["accuracy"], "f1": metrics["f1"]}, indent=2))
    print("Charts:", metrics["charts"])


if __name__ == "__main__":
    main()
