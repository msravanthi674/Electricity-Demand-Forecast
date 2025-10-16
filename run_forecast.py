"""
Streamlit app to run the fast-track 24-hour forecast for Bareilly.

Usage:
  streamlit run app.py
"""

import streamlit as st # type: ignore
from pathlib import Path
import pandas as pd # type: ignore
import io
import zipfile
import traceback
import time

# Import the pipeline functions from your project
from src.data_prep import load_and_prepare
from src.models import train_and_forecast
from src.plotting import plot_forecast_overlay, plot_horizon_mae

# helper to zip artifacts
from utils.zip_utils import zip_folder_to_bytes

ARTIFACT_DIR = Path("artifacts/fast_track")

st.set_page_config(page_title="Bareilly: 24h Demand Forecast", layout="wide")

st.title("⚡ Bareilly — 24-Hour Electricity Demand Forecast (Fast Track)")

with st.sidebar:
    st.header("Run settings")
    raw_data_input = st.text_input("Raw data path (file or folder)", value="/kaggle/input/smart-meter-data-mathura-and-bareilly")
    city = st.selectbox("City", ["Bareilly"], index=0)
    history_days = st.slider("History window (days)", min_value=3, max_value=14, value=7, step=1)
    with_weather = st.checkbox("Use Open-Meteo weather", value=True)
    forecast_origin_input = st.text_input("Forecast origin (ISO or 'auto')", value="auto")
    run_button = st.button("Run pipeline")

st.markdown("""
This app runs the cleaning → modeling → forecasting pipeline inside the notebook.  
Outputs are saved under `artifacts/fast_track/`.
""")

# status area
status = st.empty()

def show_table(df, max_rows=200):
    st.dataframe(df.head(max_rows))

def display_image(path, caption=None, width=None):
    from PIL import Image
    img = Image.open(path)
    st.image(img, caption=caption, use_column_width=(width is None))

# main action
if run_button:
    try:
        status.info("Starting pipeline...")
        t0 = time.time()
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "plots").mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "reports").mkdir(parents=True, exist_ok=True)

        # 1) load & prepare
        status.info("Loading and preparing data (this may take a moment)...")
        df_hourly, meta = load_and_prepare(
            raw_path=raw_data_input,
            city=city,
            forecast_origin=forecast_origin_input,
            history_days=history_days,
            with_weather=with_weather,
            artifact_dir=ARTIFACT_DIR
        )
        status.success(f"Data prepared. Forecast origin: {meta['forecast_origin']}")
        st.write("### Cleaned hourly snapshot")
        show_table(df_hourly.tail(72))

        # 2) train & forecast (and backtests)
        status.info("Training model and producing forecasts...")
        forecast_df, backtests = train_and_forecast(
            df_hourly=df_hourly,
            forecast_origin=meta["forecast_origin"],
            history_days=history_days,
            with_weather=with_weather,
            artifact_dir=ARTIFACT_DIR
        )

        # Save forecast CSV to artifacts
        forecast_csv = ARTIFACT_DIR / "forecast_T_plus_24.csv"
        forecast_df.to_csv(forecast_csv, index=False)

        st.write("### Forecast (T+1..T+24)")
        # ensure timestamps are timezone-aware strings
        forecast_df_display = forecast_df.copy()
        forecast_df_display['timestamp'] = pd.to_datetime(forecast_df_display['timestamp']).dt.tz_convert('Asia/Kolkata').astype(str)
        show_table(forecast_df_display, max_rows=24)

        # 3) plots
        status.info("Generating plots...")
        overlay_path = ARTIFACT_DIR / "plots" / "forecast_overlay.png"
        horizon_path = ARTIFACT_DIR / "plots" / "horizon_mae.png"
        # generate via plotting module (saves files)
        plot_forecast_overlay(df_hourly, forecast_df, meta["forecast_origin"], savepath=overlay_path)
        plot_horizon_mae(backtests, savepath=horizon_path)

        st.write("### Forecast overlay (last 3 days + T+1..T+24)")
        display_image(overlay_path, caption="Forecast overlay (IST)")

        st.write("### Horizon-wise MAE (aggregate over light backtests)")
        display_image(horizon_path, caption="Horizon-wise MAE")

        # 4) metrics table from backtests
        st.write("### Metrics (from light backtests)")
        metrics_rows = []
        from src.eval import mae, wape, smape
        for k, v in backtests.items():
            y_true = v["y_true"]
            y_pred = v["y_pred"]
            metrics_rows.append({
                "origin": k,
                "MAE": float(mae(y_true, y_pred)) if len(y_true)>0 else None,
                "WAPE": float(wape(y_true, y_pred)) if len(y_true)>0 else None,
                "sMAPE": float(smape(y_true, y_pred)) if len(y_true)>0 else None,
            })
        st.table(pd.DataFrame(metrics_rows))

        # 5) downloads: forecast csv and zipped artifacts
        st.write("### Downloads")
        with open(forecast_csv, "rb") as f:
            st.download_button("Download forecast_T_plus_24.csv", f, file_name="forecast_T_plus_24.csv")
        # zip artifacts folder to bytes and provide download
        zip_bytes = zip_folder_to_bytes(ARTIFACT_DIR)
        st.download_button("Download artifacts zip", zip_bytes, file_name="artifacts_fast_track.zip")

        elapsed = time.time() - t0
        status.success(f"Pipeline finished in {elapsed:.1f}s — artifacts written to {ARTIFACT_DIR}")
    except Exception as e:
        status.error("Pipeline failed — see error below")
        st.error(traceback.format_exc())
