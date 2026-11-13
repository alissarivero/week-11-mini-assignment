"""Scheduled Airflow DAG: ingest two related tables, clean in parallel, merge, load, analyze."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task, task_group

from pipeline.analyze import run_analysis
from pipeline.config import BIKE_RAW, WEATHER_RAW
from pipeline.ingest import ingest_bikes, ingest_weather
from pipeline.load import load_merged
from pipeline.transform import clean_bikes, clean_weather, merge_datasets

DEFAULT_ARGS = {
    "owner": "week11",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="bike_demand_etl",
    description="Ingest bike rentals + weather, merge into Postgres, train a demand classifier.",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["week11", "etl", "ml"],
)
def bike_demand_etl():
    @task_group(group_id="ingest")
    def ingest():
        @task(task_id="ingest_bikes")
        def _bikes() -> str:
            return ingest_bikes(BIKE_RAW)

        @task(task_id="ingest_weather")
        def _weather() -> str:
            return ingest_weather(WEATHER_RAW)

        return _bikes(), _weather()

    @task_group(group_id="clean_bikes")
    def bikes_tg(path: str) -> str:
        @task(task_id="strip_and_validate")
        def _clean(p: str) -> str:
            return clean_bikes(p)

        return _clean(path)

    @task_group(group_id="clean_weather")
    def weather_tg(path: str) -> str:
        @task(task_id="interpolate_and_clip")
        def _clean(p: str) -> str:
            return clean_weather(p)

        return _clean(path)

    @task
    def merge(bike_path: str, weather_path: str) -> str:
        return merge_datasets(bike_path, weather_path)

    @task
    def load(merged_path: str) -> dict:
        return load_merged(merged_path)

    @task
    def analyze(load_info: dict) -> dict:
        return run_analysis()

    bike_raw, weather_raw = ingest()
    bike_clean = bikes_tg(bike_raw)
    weather_clean = weather_tg(weather_raw)
    merged = merge(bike_clean, weather_clean)
    loaded = load(merged)
    analyze(loaded)


bike_demand_etl()
