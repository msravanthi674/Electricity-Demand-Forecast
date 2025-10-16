"""
Create the two required plots:
1) Last 3 days actuals with final 24h forecast overlay
2) Horizon-wise MAE for horizons 1..24 (from backtests)
"""

import matplotlib.pyplot as plt
import pandas as pd # type: ignore
import numpy as np

def plot_forecast_overlay(df_hourly, forecast_df, forecast_origin, savepath=None):
    # select last 3 days of actuals up to forecast_origin
    end = forecast_origin
    start = end - pd.Timedelta(days=3) + pd.Timedelta(hours=1)
    sub = df_hourly.loc[start:end].copy()
    plt.figure(figsize=(12,4))
    plt.plot(sub.index, sub['hourly_kwh'], label='Actual (last 3 days)')
    # overlay forecast
    fts = forecast_df.set_index(pd.to_datetime(forecast_df['timestamp'])).sort_index()
    plt.plot(fts.index, fts['yhat'], label='Forecast (T+1..T+24)', linewidth=2)
    plt.fill_between(fts.index, fts['y_p10'], fts['y_p90'], alpha=0.2, label='p10-p90')
    plt.axvline(forecast_origin, color='k', linestyle='--', label='Forecast origin')
    plt.xlabel("Time (IST)")
    plt.ylabel("Hourly kWh")
    plt.title("Last 3 days: actuals with 24-hour forecast overlay")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.close()

def plot_horizon_mae(backtests, savepath=None):
    """
    backtests: dict keys origin_x => dict with y_true and y_pred arrays for horizons 1..24
    We'll compute MAE for each horizon across available backtests.
    """
    # backtests stored per origin with arrays of length up to 24 (but may be shorter)
    # Build per-horizon errors
    horizon_errors = {h: [] for h in range(1,25)}
    for k,v in backtests.items():
        ytrue = v["y_true"]
        ypred = v["y_pred"]
        # align horizon: assume entries correspond to h=1..len
        for i in range(min(24, len(ytrue))):
            horizon_errors[i+1].append(abs(ytrue[i] - ypred[i]))
    horizons = sorted(horizon_errors.keys())
    mae_vals = [np.mean(horizon_errors[h]) if horizon_errors[h] else np.nan for h in horizons]
    plt.figure(figsize=(10,4))
    plt.plot(horizons, mae_vals, marker='o')
    plt.xlabel("Horizon (hours)")
    plt.ylabel("MAE (kWh)")
    plt.title("Horizon-wise MAE (aggregate over backtests)")
    plt.grid(True)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.close()
