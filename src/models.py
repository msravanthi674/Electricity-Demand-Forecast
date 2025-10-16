"""
src/models.py

Robust Ridge forecasting pipeline for the fast-track forecast project.

Key fix:
- iterative_forecast(): ensure tmp is reindexed to include working.index and the forecast timestamp
  before assigning values, avoiding KeyError when assigning by .loc.

Exports: train_and_forecast, iterative_forecast, seasonal_naive_forecast
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd # type: ignore
import numpy as np
from sklearn.linear_model import Ridge # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from joblib import dump, load # type: ignore

from src.features import build_features

__all__ = ["train_and_forecast", "iterative_forecast", "seasonal_naive_forecast"]

MODEL_FILENAME = "model_ridge.joblib"


def prepare_train_matrix(df_hourly: pd.DataFrame, forecast_origin: pd.Timestamp,
                         history_days: int = 7, include_temp: bool = False
                         ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare X, y for training using the window [T-history_days, ..., T-1]
    """
    end = forecast_origin - pd.Timedelta(hours=1)
    start = end - pd.Timedelta(days=history_days) + pd.Timedelta(hours=1)
    train_df = df_hourly.loc[start:end].copy()
    X, df_feats = build_features(train_df, include_temp)
    y = train_df["hourly_kwh"]
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y, df_feats


def fit_ridge(X: pd.DataFrame, y: pd.Series, artifact_dir: Optional[Path] = None) -> Pipeline:
    """
    Fit a Ridge pipeline (StandardScaler + Ridge), save model to artifact_dir, and return pipeline.
    """
    if artifact_dir is None:
        artifact_dir = Path("artifacts/fast_track")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / MODEL_FILENAME

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model.fit(X, y)
    dump(model, model_path)
    return model


def seasonal_naive_forecast(df_hourly: pd.DataFrame, forecast_origin: pd.Timestamp) -> pd.DataFrame:
    """
    Seasonal naive baseline: for h=1..24 predict value at T+h-24.
    Returns DataFrame with columns ['timestamp','yhat'] (timestamp tz-aware).
    """
    preds = []
    for h in range(1, 25):
        t = forecast_origin + pd.Timedelta(hours=h)
        ref = t - pd.Timedelta(hours=24)
        val = df_hourly["hourly_kwh"].get(ref, np.nan)
        preds.append({"timestamp": t, "yhat": float(val) if not pd.isna(val) else float("nan")})
    return pd.DataFrame(preds)


def iterative_forecast(model_pipeline: Pipeline, df_hourly: pd.DataFrame,
                       forecast_origin: pd.Timestamp, include_temp: bool = False) -> pd.DataFrame:
    """
    Iteratively produce multi-step forecasts for horizons 1..24 using model_pipeline.
    For lags beyond available actuals, use previously-predicted values (stored in 'working').

    Robustness fix: ensure tmp is reindexed to include working.index and the forecast timestamp t
    before any assignment, avoiding KeyError when working.index contains timestamps not present in tmp.
    """
    results: List[Dict[str, Any]] = []
    # copy working series so we can add predicted future values for subsequent lags
    working = df_hourly["hourly_kwh"].copy()

    for h in range(1, 25):
        t = forecast_origin + pd.Timedelta(hours=h)

        # create a temporary df that includes all indices from df_hourly and working, plus t
        # this ensures assignments using working.index won't fail
        all_idx = df_hourly.index.union(working.index)
        if t not in all_idx:
            all_idx = all_idx.union(pd.DatetimeIndex([t]))
        tmp = df_hourly.reindex(all_idx).sort_index()

        # ensure column exists
        if "hourly_kwh" not in tmp.columns:
            tmp["hourly_kwh"] = np.nan

        # populate tmp['hourly_kwh'] from working where available (reindex working to tmp.index)
        tmp.loc[working.index, "hourly_kwh"] = working.reindex(working.index).values

        # build features
        X_all, _ = build_features(tmp, include_temp=include_temp)

        # ensure we have a row for t
        if t not in X_all.index:
            # create a fallback row filled by nearest values
            # use last available row as template then set index to t
            fallback = X_all.ffill().bfill().iloc[[0]].copy()
            fallback.index = pd.DatetimeIndex([t]).tz_localize(tmp.index.tz)
            X_row = fallback
        else:
            X_row = X_all.loc[[t]]

        # try filling temperature from df_hourly if include_temp
        if include_temp and "temperature_2m" in df_hourly.columns and "temp" in X_row.columns:
            if pd.isna(X_row["temp"].iloc[0]):
                try:
                    X_row["temp"] = df_hourly["temperature_2m"].loc[t]
                except Exception:
                    pass

        # handle any remaining NaNs
        if X_row.isnull().any(axis=None):
            X_row = X_row.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # predict
        yhat = model_pipeline.predict(X_row)[0]
        results.append({"timestamp": t, "yhat": float(yhat)})

        # append predicted value to working so next horizon can use it as lag
        working.loc[t] = yhat

    return pd.DataFrame(results)


def compute_residual_quantiles(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                               qlist=(0.10, 0.50, 0.90)) -> Dict[str, float]:
    preds = model.predict(X_train)
    residuals = y_train.values - preds
    qs = np.quantile(residuals, list(qlist))
    return {f"q{int(q*100)}": float(val) for q, val in zip(qlist, qs)}


def train_and_forecast(df_hourly: pd.DataFrame, forecast_origin: pd.Timestamp,
                       history_days: int = 7, with_weather: bool = True,
                       artifact_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end train + forecast wrapper expected by run_forecast.py and Streamlit app.

    Returns:
      - forecast_df: DataFrame with ['timestamp','yhat','y_p10','y_p50','y_p90']
      - backtests: dict of backtest results keyed by origin string.
    """
    if artifact_dir is None:
        artifact_dir = Path("artifacts/fast_track")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # baseline (not used further here but could be saved)
    _ = seasonal_naive_forecast(df_hourly, forecast_origin)

    # prepare training data
    X_train, y_train, _ = prepare_train_matrix(df_hourly, forecast_origin, history_days, include_temp=with_weather)

    # fit model
    model = fit_ridge(X_train, y_train, artifact_dir=artifact_dir)

    # produce iterative multi-step predictions
    pred_df = iterative_forecast(model, df_hourly, forecast_origin, include_temp=with_weather)

    # compute quantiles from residuals (on training residuals)
    q = compute_residual_quantiles(model, X_train, y_train, qlist=(0.10, 0.50, 0.90))
    pred_df["y_p10"] = pred_df["yhat"] + q["q10"]
    pred_df["y_p50"] = pred_df["yhat"] + q["q50"]
    pred_df["y_p90"] = pred_df["yhat"] + q["q90"]
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

    # Light backtests for origins T, T-1, T-2 days
    backtests: Dict[str, Any] = {}
    for d in (0, 1, 2):
        origin = forecast_origin - pd.Timedelta(days=d)
        key = f"origin_{origin.strftime('%Y-%m-%dT%H')}"
        try:
            X_tr, y_tr, _ = prepare_train_matrix(df_hourly, origin, history_days, include_temp=with_weather)
            mod = fit_ridge(X_tr, y_tr, artifact_dir=artifact_dir)
            pred_bt = iterative_forecast(mod, df_hourly, origin, include_temp=with_weather)
            # align preds with actuals
            actuals = []
            preds = []
            for _, row in pred_bt.iterrows():
                ts = row["timestamp"]
                if ts in df_hourly.index:
                    actuals.append(float(df_hourly.loc[ts, "hourly_kwh"]))
                    preds.append(row["yhat"])
            backtests[key] = {"y_true": actuals, "y_pred": preds, "origin": origin}
        except Exception as e:
            backtests[key] = {"y_true": [], "y_pred": [], "origin": origin, "error": str(e)}

    return pred_df[["timestamp", "yhat", "y_p10", "y_p50", "y_p90"]], backtests


# -----------------------
# Quick debug helper (optional)
# -----------------------
if __name__ == "__main__":
    # Quick local check for dataset path (edit as needed)
    p = Path(r"C:\Users\Admin\OneDrive\Documents\fast_track_forecast\forecast")
    print("exists:", p.exists())
    if p.exists():
        print("csvs:", list(p.glob("**/*.csv"))[:10])
    else:
        print("path not found; please point to a folder with CSV(s) or a zip")
