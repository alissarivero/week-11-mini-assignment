"""Load the merged table into PostgreSQL (or SQLite for local runs)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from pipeline.config import ANALYTICS_DB_URI, MERGED_CSV, TABLE_NAME


def get_engine(uri: str | None = None):
    return create_engine(uri or ANALYTICS_DB_URI)


def load_merged(csv_path: str | Path | None = None, uri: str | None = None) -> dict:
    path = Path(csv_path) if csv_path else MERGED_CSV
    df = pd.read_csv(path, parse_dates=["ride_date"])
    engine = get_engine(uri)
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    return {"table": TABLE_NAME, "rows": int(n), "uri": uri or ANALYTICS_DB_URI}


def read_merged(uri: str | None = None) -> pd.DataFrame:
    engine = get_engine(uri)
    return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine, parse_dates=["ride_date"])
