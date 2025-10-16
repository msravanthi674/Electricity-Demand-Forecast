"""
app.py — Streamlit App for Bareilly Electricity Demand Forecast

Upload the dataset (CSV or ZIP) and generate a 24-hour forecast
using smart-meter + weather data.

Author: M. Sravanthi
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import time
import matplotlib.pyplot as plt

from src.data_prep import load_and_prepare
from src.models import train_and_forecast

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(
    page_title="⚡ Bareilly Electricity Forecast",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Bareilly — 24-Hour Electricity Demand Forecast (Fast Track)")
st.caption("AI-powered forecasting using smart-meter + weather data (Open-Meteo API)")

st.sidebar.header("⚙️ Run Settings")

# ------------------------
# File Upload Section
# ------------------------
uploaded = st.sidebar.file_uploader("Upload dataset (CSV or ZIP)", type=["csv", "zip"])
if uploaded:
    tmp_dir = Path(tempfile.mkdtemp())
    file_path = tmp_dir / uploaded.name
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    raw_data_path = str(file_path)
    st.sidebar.success(f"✅ Uploaded file saved to {raw_data_path}")
else:
    raw_data_path = None
    st.sidebar.info("Please upload the Bareilly dataset (CSV or ZIP) to run the forecast.")

# ------------------------
# Additional Inputs
# ------------------------
city = st.sidebar.selectbox("City", ["Bareilly", "Mathura"], index=0)
history_window_days = st.sidebar.slider("History window (days)", 3, 14, 7)
forecast_origin = st.sidebar.text_input("Forecast origin (ISO or 'auto')", "auto")
with_weather = st.sidebar.checkbox("Include weather (Open-Meteo)", value=True)

# ------------------------
# Run Button
# ------------------------
if st.sidebar.button("🚀 Run Forecast") and raw_data_path:
    start_time = time.time()
    with st.spinner("Running forecast pipeline..."):
        try:
            df_hourly, meta = load_and_prepare(
                raw_data_path,
                city=city,
                forecast_origin=forecast_origin,
                history_days=history_window_days,
                with_weather=with_weather,
            )

            forecast_df, backtests = train_and_forecast(
                df_hourly,
                meta["forecast_origin"],
                history_days=history_window_days,
                with_weather=with_weather,
            )

            elapsed = time.time() - start_time
            st.success(f"✅ Pipeline finished in {elapsed:.1f}s — artifacts written to artifacts/fast_track/")

            # ------------------------
            # Display Results
            # ------------------------
            st.subheader("📊 Cleaned Hourly Snapshot")
            st.dataframe(df_hourly.tail(10), use_container_width=True)

            st.subheader("🔮 Forecast (T+1..T+24)")
            st.dataframe(forecast_df.head(24), use_container_width=True)

            # Plot: forecast overlay
            st.subheader("📈 Forecast Overlay (last 3 days + next 24h)")
            fig, ax = plt.subplots(figsize=(10, 4))
            last = df_hourly.tail(72)
            ax.plot(last.index, last["hourly_kwh"], label="Actual", color="tab:blue")
            ax.plot(forecast_df["timestamp"], forecast_df["yhat"], label="Forecast", color="tab:orange")
            ax.fill_between(
                forecast_df["timestamp"],
                forecast_df["y_p10"],
                forecast_df["y_p90"],
                color="tab:orange",
                alpha=0.2,
                label="P10–P90"
            )
            ax.set_title("Forecast Overlay (IST)")
            ax.legend()
            st.pyplot(fig, use_container_width=True)

            # Plot: horizon-wise MAE (aggregate over backtests)
            st.subheader("📉 Horizon-wise MAE (from light backtests)")
            maes = []
            for k, v in backtests.items():
                if "y_true" in v and len(v["y_true"]) > 0:
                    y_true = pd.Series(v["y_true"])
                    y_pred = pd.Series(v["y_pred"])
                    mae = (y_true - y_pred).abs().mean()
                    maes.append({"origin": k, "MAE": mae})
            if maes:
                mae_df = pd.DataFrame(maes)
                st.bar_chart(mae_df.set_index("origin"), use_container_width=True)
            else:
                st.info("No valid backtest data for MAE plot.")

            # Metrics table (simple)
            st.subheader("📋 Metrics (from light backtests)")
            metrics = []
            for k, v in backtests.items():
                if "y_true" in v and len(v["y_true"]) > 0:
                    y_true = pd.Series(v["y_true"])
                    y_pred = pd.Series(v["y_pred"])
                    mae = (y_true - y_pred).abs().mean()
                    wape = mae / y_true.mean() * 100 if y_true.mean() != 0 else float("nan")
                    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
                    metrics.append({"origin": k, "MAE": mae, "WAPE": wape, "sMAPE": smape})
            if metrics:
                st.dataframe(pd.DataFrame(metrics).round(3), use_container_width=True)
            else:
                st.warning("Metrics unavailable — possibly due to missing actuals.")

        except Exception as e:
            st.error(f"❌ Pipeline failed — {e}")

elif not uploaded:
    st.info("👆 Upload a dataset on the sidebar to begin.")
else:
    st.warning("Please click 'Run Forecast' to start.")
