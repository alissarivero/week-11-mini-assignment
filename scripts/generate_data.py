"""Synthesize two related daily datasets: station bike rentals and city weather."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.config import BIKE_RAW, WEATHER_RAW, ensure_dirs  # noqa: E402

STATIONS = [
    ("ST-01", "Riverfront Dock"),
    ("ST-02", "University Gate"),
    ("ST-03", "Market Square"),
    ("ST-04", "Lakeside Trail"),
    ("ST-05", "Central Station"),
]
STATION_BASE = {
    "ST-01": 95,
    "ST-02": 140,
    "ST-03": 180,
    "ST-04": 70,
    "ST-05": 210,
}


def main(seed: int = 42) -> None:
    ensure_dirs()
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

    doy = dates.day_of_year.to_numpy()
    seasonal = 12 + 14 * np.sin(2 * np.pi * (doy - 80) / 365)
    temp_c = seasonal + rng.normal(0, 3.2, size=len(dates))
    humidity = np.clip(62 - 0.6 * (temp_c - 12) + rng.normal(0, 8, size=len(dates)), 20, 98)
    wind_kph = np.clip(rng.gamma(2.2, 4.5, size=len(dates)), 1, 55)
    precip_mm = np.where(rng.random(len(dates)) < 0.28, rng.exponential(4.5, size=len(dates)), 0)
    weather_code = np.where(
        precip_mm > 8,
        "storm",
        np.where(precip_mm > 0.5, "rain", np.where(temp_c < 2, "cold", "clear")),
    )

    weather = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temp_c": np.round(temp_c, 2),
            "humidity": np.round(humidity, 1),
            "wind_kph": np.round(wind_kph, 1),
            "precip_mm": np.round(precip_mm, 2),
            "weather_code": weather_code,
        }
    )

    rows = []
    for station_id, station_name in STATIONS:
        weekend = (dates.day_of_week >= 5).astype(int)
        demand = (
            STATION_BASE[station_id]
            + 5.5 * temp_c
            - 8.0 * precip_mm
            - 0.9 * wind_kph
            - 0.12 * humidity
            + 28 * weekend
            + rng.normal(0, 36, size=len(dates))
        )
        demand = np.clip(np.round(demand), 0, None).astype(int)
        member_share = np.clip(0.58 + 0.08 * weekend + rng.normal(0, 0.05, size=len(dates)), 0.35, 0.9)
        member = np.round(demand * member_share).astype(int)
        casual = demand - member
        for i, day in enumerate(dates):
            rows.append(
                {
                    "ride_date": day.strftime("%Y-%m-%d"),
                    "station_id": station_id,
                    "station_name": station_name,
                    "rentals": int(demand[i]),
                    "casual_rides": int(casual[i]),
                    "member_rides": int(member[i]),
                }
            )

    bikes = pd.DataFrame(rows)
    # Inject a few dirty values the cleaning tasks should handle.
    bikes.loc[10, "rentals"] = -3
    bikes.loc[25, "station_name"] = " university gate "
    weather.loc[4, "humidity"] = 140
    weather.loc[18, "temp_c"] = np.nan

    bikes.to_csv(BIKE_RAW, index=False)
    weather.to_csv(WEATHER_RAW, index=False)
    print(f"Wrote {len(bikes)} bike rows -> {BIKE_RAW}")
    print(f"Wrote {len(weather)} weather rows -> {WEATHER_RAW}")


if __name__ == "__main__":
    main()
