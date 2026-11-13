"""Clean each source table and merge them on calendar date."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import BIKE_CLEAN, DATA_PROCESSED, MERGED_CSV, WEATHER_CLEAN, ensure_dirs


def clean_bikes(src: str | Path) -> str:
    ensure_dirs()
    df = pd.read_csv(src)
    df["ride_date"] = pd.to_datetime(df["ride_date"], errors="coerce")
    df["station_id"] = df["station_id"].astype(str).str.strip().str.upper()
    df["station_name"] = df["station_name"].astype(str).str.strip().str.title()
    for col in ("rentals", "casual_rides", "member_rides"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ride_date", "station_id", "rentals"])
    df = df[df["rentals"] >= 0]
    df["casual_rides"] = df["casual_rides"].clip(lower=0).fillna(0).astype(int)
    df["member_rides"] = df["member_rides"].clip(lower=0).fillna(0).astype(int)
    df["rentals"] = df["rentals"].astype(int)
    df = df.drop_duplicates(subset=["ride_date", "station_id"])
    df.to_csv(BIKE_CLEAN, index=False)
    return str(BIKE_CLEAN)


def clean_weather(src: str | Path) -> str:
    ensure_dirs()
    df = pd.read_csv(src)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("temp_c", "humidity", "wind_kph", "precip_mm"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["humidity"] = df["humidity"].clip(upper=100)
    df = df.sort_values("date")
    df[["temp_c", "humidity", "wind_kph"]] = df[["temp_c", "humidity", "wind_kph"]].interpolate(
        limit_direction="both"
    )
    df["precip_mm"] = df["precip_mm"].fillna(0)
    df["weather_code"] = df["weather_code"].fillna("unknown").str.lower().str.strip()
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date"])
    df.to_csv(WEATHER_CLEAN, index=False)
    return str(WEATHER_CLEAN)


def merge_datasets(bike_path: str | Path, weather_path: str | Path) -> str:
    ensure_dirs()
    bikes = pd.read_csv(bike_path, parse_dates=["ride_date"])
    weather = pd.read_csv(weather_path, parse_dates=["date"])
    merged = bikes.merge(weather, left_on="ride_date", right_on="date", how="inner")
    merged = merged.drop(columns=["date"])
    merged["is_weekend"] = (merged["ride_date"].dt.day_of_week >= 5).astype(int)
    merged["month"] = merged["ride_date"].dt.month
    merged["day_of_week"] = merged["ride_date"].dt.day_of_week
    merged.to_csv(MERGED_CSV, index=False)
    qc = DATA_PROCESSED / "merge_qc.csv"
    summary = pd.DataFrame(
        {
            "metric": ["rows", "stations", "days", "nulls"],
            "value": [
                len(merged),
                merged["station_id"].nunique(),
                merged["ride_date"].nunique(),
                int(merged.isna().sum().sum()),
            ],
        }
    )
    summary.to_csv(qc, index=False)
    return str(MERGED_CSV)
