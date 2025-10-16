"""
Load raw smart-meter CSV(s), filter to city, aggregate 3-minute to hourly,
handle missing timestamps, impute small gaps, cap outliers, and optionally fetch weather.
Updated: safer timestamp parsing to avoid applying dateutil.parser to pandas.Timestamp objects.
"""

import pandas as pd # type: ignore
import numpy as np
import os
from pathlib import Path
import glob
import requests
from dateutil import parser
import pytz # type: ignore
import zipfile
from typing import Optional, Tuple

# Bareilly approx coordinates (used for Open-Meteo)
COORDS = {
    "Bareilly": {"lat": 28.3670, "lon": 79.4304},
    "Mathura": {"lat": 27.4924, "lon": 77.6737}
}


def find_csvs(path: str):
    p = Path(path)
    if p.is_dir():
        return list(map(str, p.glob("**/*.csv")))
    if p.is_file():
        if p.suffix.lower() == ".zip":
            # extract to a directory next to the zip file
            extract_dir = p.parent / (p.stem + "_unzipped")
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(extract_dir)
            return list(map(str, extract_dir.glob("**/*.csv")))
        if p.suffix.lower() == ".csv":
            return [str(p)]
    raise FileNotFoundError(f"No CSVs found at {path}")


def infer_column(cols, candidates):
    for c in candidates:
        for col in cols:
            if col.lower() == c.lower():
                return col
    # try contains
    for c in candidates:
        for col in cols:
            if c.lower() in col.lower():
                return col
    return None


def load_raw_csvs(csv_paths):
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df["_source_file"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"Failed reading {p}: {e}")
    if not dfs:
        raise RuntimeError("No CSVs loaded")
    return pd.concat(dfs, ignore_index=True, sort=False)


def parse_timestamp_column(df):
    candidates = ["timestamp", "time", "datetime", "date", "read_time", "starttime", "ts"]
    col = infer_column(df.columns, candidates)
    if col is None:
        # try any column with datetime-like values
        for c in df.columns:
            if df[c].dtype == object:
                sample = df[c].dropna().astype(str).head(5).tolist()
                try:
                    for s in sample:
                        parser.parse(s)
                    return c
                except Exception:
                    continue
        raise RuntimeError("Could not infer timestamp column; please pass a CSV with a datetime column")
    return col


def infer_value_column(df):
    candidates = ["energy", "energy_kwh", "kwh", "consumption", "value", "meter_reading", "reading"]
    col = infer_column(df.columns, candidates)
    if col is None:
        # pick numeric column excluding timestamp
        for c in df.select_dtypes(include=[np.number]).columns:
            return c
    return col


def infer_city_column(df):
    candidates = ["city", "location", "site", "district", "area", "place", "name"]
    return infer_column(df.columns, candidates)


def ensure_datetime_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Robust conversion of ts_col to datetime. Handles mixed types (strings, pd.Timestamp, etc).
    """
    df = df.copy()
    # Try vectorized conversion first
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    # If some rows remain NaT, try parsing those individually (but only non-Timestamp values)
    if df[ts_col].isnull().any():
        def safe_parse(x):
            if pd.isna(x):
                return pd.NaT
            # if it's already a Timestamp/Datetime, return as-is
            if isinstance(x, pd.Timestamp):
                return x
            try:
                # convert to str first to protect parser
                return parser.parse(str(x))
            except Exception:
                return pd.NaT
        # Apply only to entries that are NaT (attempt to recover), keeping others
        mask_nat = df[ts_col].isna()
        if mask_nat.any():
            parsed = df.loc[mask_nat, ts_col].apply(safe_parse)
            df.loc[mask_nat, ts_col] = parsed
    return df


def resample_to_hour(df, ts_col, value_col, city_col, city_name):
    df = df.copy()
    # Filter by city if available
    if city_col is not None and city_col in df.columns and city_name is not None:
        mask = df[city_col].astype(str).str.contains(city_name, case=False, na=False)
        df = df.loc[mask]
        if df.empty:
            print(f"Warning: no rows found for city '{city_name}' using city_col '{city_col}' -- proceeding with full dataset")
    # unify tz: assume incoming timestamps are naive -> treat as UTC then convert to IST
    df = df.set_index(pd.DatetimeIndex(df[ts_col]))
    # If tz-naive, assume UTC (common for Kaggle). Then convert to IST
    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        except Exception:
            # if localization fails, try to coerce to naive then localize
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')

    # If multiple meters per hour, sum across them
    if value_col not in df.columns:
        raise RuntimeError(f"Value column '{value_col}' not found in dataframe")
    hourly = df[[value_col]].groupby(pd.Grouper(freq='H')).sum()
    hourly = hourly.rename(columns={value_col: "hourly_kwh"})
    # Ensure continuous hourly index
    idx = pd.date_range(start=hourly.index.min(), end=hourly.index.max(), freq='H', tz='Asia/Kolkata')
    hourly = hourly.reindex(idx)
    return hourly


def impute_and_cap(hourly, max_gap_hours=2, lower_pct=1, upper_pct=99):
    df = hourly.copy()
    # count original missing
    n_missing_before = df['hourly_kwh'].isna().sum()
    # small gap interpolation
    df['hourly_kwh'] = df['hourly_kwh'].interpolate(method='time', limit=max_gap_hours)
    df['hourly_kwh'] = df['hourly_kwh'].fillna(method='ffill').fillna(method='bfill')
    n_missing_after = df['hourly_kwh'].isna().sum()
    # cap outliers
    low = np.nanpercentile(df['hourly_kwh'], lower_pct)
    high = np.nanpercentile(df['hourly_kwh'], upper_pct)
    # create clipped version from original (so we can count clipped values)
    clipped = df['hourly_kwh'].clip(lower=low, upper=high)
    clip_counts = ((hourly['hourly_kwh'] != clipped) & hourly['hourly_kwh'].notna()).sum() if 'hourly_kwh' in hourly.columns else 0
    df['hourly_kwh'] = clipped
    audit = {"missing_before": int(n_missing_before), "missing_after": int(n_missing_after),
             "capped_to_lower": float(low), "capped_to_upper": float(high),
             "num_clipped": int(clip_counts)}
    return df, audit


def fetch_open_meteo(lat, lon, start_iso, end_iso):
    """
    Fetch hourly temperature forecast from Open-Meteo. Expects ISO datetimes in UTC or TZ-aware.
    We'll request hourly temperature_2m between start and end (inclusive).
    """
    # ensure start_iso/end_iso are timezone-aware pandas timestamps
    start = pd.to_datetime(start_iso)
    end = pd.to_datetime(end_iso)
    # convert to UTC strings for the API
    start_date = start.tz_convert('UTC').strftime('%Y-%m-%d')
    end_date = end.tz_convert('UTC').strftime('%Y-%m-%d')
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&hourly=temperature_2m"
           f"&start_date={start_date}"
           f"&end_date={end_date}"
           f"&timezone=UTC")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    times = pd.to_datetime(j["hourly"]["time"]).tz_localize('UTC')
    temps = j["hourly"]["temperature_2m"]
    df = pd.DataFrame({"temperature_2m": temps}, index=times)
    # convert to IST
    df.index = df.index.tz_convert('Asia/Kolkata')
    return df


def load_and_prepare(raw_path: str, city: str = "Bareilly", forecast_origin: str = "auto",
                     history_days: int = 7, with_weather: bool = True,
                     artifact_dir: Optional[Path] = Path("artifacts/fast_track")) -> Tuple[pd.DataFrame, dict]:
    csvs = find_csvs(raw_path)
    print(f"Found {len(csvs)} CSV files; loading...")
    raw = load_raw_csvs(csvs)
    ts_col = parse_timestamp_column(raw)
    val_col = infer_value_column(raw)
    city_col = infer_city_column(raw)
    print(f"Inferred timestamp='{ts_col}', value='{val_col}', city_col='{city_col}'")
    raw = ensure_datetime_index(raw, ts_col)

    hourly = resample_to_hour(raw, ts_col, val_col, city_col, city)
    hourly_clean, audit = impute_and_cap(hourly)

    # forecast origin
    if forecast_origin == "auto":
        forecast_origin = hourly_clean.index.max()
    else:
        forecast_origin = pd.to_datetime(forecast_origin)
        # make timezone-aware IST if naive
        if forecast_origin.tzinfo is None:
            forecast_origin = forecast_origin.tz_localize('Asia/Kolkata')
        else:
            forecast_origin = forecast_origin.tz_convert('Asia/Kolkata')

    # optionally fetch weather for T+1..T+24
    weather_df = None
    if with_weather:
        coords = COORDS.get(city, COORDS["Bareilly"])
        start = forecast_origin + pd.Timedelta(hours=1)
        end = forecast_origin + pd.Timedelta(hours=24)
        try:
            weather_df = fetch_open_meteo(coords["lat"], coords["lon"], start, end)
            # filter to exactly the 24 hours
            weather_df = weather_df.reindex(pd.date_range(start=start, end=end, freq='H', tz='Asia/Kolkata'))
        except Exception as e:
            print(f"Warning: failed to fetch weather: {e}")
            weather_df = None

    meta = {"forecast_origin": forecast_origin, "audit": audit, "ts_col": ts_col, "val_col": val_col, "city_col": city_col, "weather": bool(weather_df)}
    # merge weather into hourly_clean for convenience (future temps will be used separately)
    if weather_df is not None:
        hourly_clean = hourly_clean.join(weather_df, how='left')

    return hourly_clean, meta


# Optional: quick check helper if running this file directly
if __name__ == "__main__":
    # Edit this path as needed to test locally
    test_p = Path(r"C:\Users\Admin\OneDrive\Documents\fast_track_forecast\forecast")
    print("Path exists:", test_p.exists())
    if test_p.exists():
        print("CSV samples:", list(test_p.glob("**/*.csv"))[:10])
    else:
        print("Test path does not exist; please point to a folder containing CSV(s) or a zip.")
