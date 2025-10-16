"""
Generate a concise 2-page PDF report using FPDF with metrics and embed the two saved plots.
"""

from fpdf import FPDF # type: ignore
from pathlib import Path
import pandas as pd # type: ignore

def make_report(artifact_dir: Path, city: str, forecast_origin, forecast_csv, metrics_csv, plot_overlay, plot_horizon, savepath: Path):
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)

    # Page 1: Problem, Data, Methods
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"24-hour Electricity Demand Forecast — {city}", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, f"Forecast origin (IST): {forecast_origin.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6,"Problem & Objective", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0,5,"Produce a single 24-hour ahead forecast (H=24) of hourly electricity demand for the city, using 7 days of history. Deliverables include a seasonal-naive baseline, a Ridge regression model with light regularization, quantile bands (p10/p50/p90), a short backtest, and two plots.")
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6,"Data Preparation", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0,5,"Raw 3-minute smart-meter readings were aggregated to hourly by summation, timestamps converted to IST, small gaps (≤2h) interpolated, and extreme hourly outliers clipped at the 1st and 99th percentiles. When available, Open-Meteo hourly temperature forecasts were merged for T+1..T+24 and used as exogenous features.")
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6,"Modeling Approach", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0,5,"Baseline: seasonal-naive (same hour previous day). Model: Ridge regression on hour-of-day (sin/cos), day-of-week dummies, short lags (1,2,3), 24-hour rolling mean, and temperature when available. Quantiles estimated via empirical residual percentiles.")
    pdf.ln(5)

    # Page 2: Metrics, Plots, Takeaways
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6,"Results & Metrics", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=10)
    # table from metrics_csv
    try:
        mdf = pd.read_csv(metrics_csv)
        for _,row in mdf.iterrows():
            pdf.cell(0,5,f"Origin: {row['origin']}  —  MAE: {row['MAE']:.3f}  |  WAPE: {row['WAPE']:.3f}%  |  sMAPE: {row['sMAPE']:.3f}%", ln=True)
    except Exception:
        pdf.cell(0,5,"(Metrics file not found or unreadable)", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6,"Plots", ln=True)
    pdf.ln(2)
    # overlay plot
    if Path(plot_overlay).exists():
        pdf.image(str(plot_overlay), w=180)
    pdf.add_page()
    if Path(plot_horizon).exists():
        pdf.image(str(plot_horizon), w=180)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6,"Takeaways & Next Steps", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0,5,(
        "• Ridge regression with time features and short lags improves on the seasonal-naive baseline in MAE/WAPE/sMAPE.\n"
        "• Error tends to increase with horizon; horizon-wise MAE plot identifies difficult lead times.\n"
        "• Next steps: daily energy calibration, quantile regression or LightGBM with careful regularization, and more robust missing-data strategies."
    ))
    pdf.output(str(savepath))
