"""
Streamlit prototype for the IIIT Lucknow Climate Intelligence Challenge 2026
(Aether AI/ML Club) -- 10-day Wet-Bulb Temperature (WBT) heat-risk forecaster.

Run locally:
    streamlit run app/app.py

Deploy: push this repo to GitHub and point Streamlit Community Cloud at
app/app.py (see README.md).
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
MODEL_DIR = os.path.join(os.path.dirname(APP_DIR), "model")

DAY_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
RISK_COLORS = {"Safe": "#2E7D32", "Caution": "#F9A825", "Danger": "#EF6C00", "Fatal": "#C62828"}
RISK_ORDER = ["Safe", "Caution", "Danger", "Fatal"]

st.set_page_config(page_title="UP Heat-Risk Forecaster", page_icon="\U0001F321️", layout="wide")


@st.cache_data
def load_predictions():
    df = pd.read_parquet(os.path.join(DATA_DIR, "predictions.parquet"))
    return df


@st.cache_data
def load_districts():
    return pd.read_csv(os.path.join(DATA_DIR, "districts.csv"))


@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def risk_badge(level):
    color = RISK_COLORS.get(level, "#999")
    return f'<span style="background:{color};color:white;padding:3px 12px;border-radius:12px;font-weight:600;">{level}</span>'


st.title("\U0001F321️ Uttar Pradesh Heat-Risk Forecaster")
st.caption(
    "10-day-ahead Wet-Bulb Temperature (WBT) forecasts across 125 districts of UP & neighboring "
    "regions — built for the IIIT Lucknow Climate Intelligence Challenge 2026 (Aether)."
)

predictions = None
try:
    predictions = load_predictions()
    districts = load_districts()
except FileNotFoundError:
    st.error(
        "No scored predictions found yet. Run `py model/train_model.py` then "
        "`py model/score_test.py` from the project root to generate "
        "`app/data/predictions.parquet` before launching this app."
    )
    st.stop()

metrics = load_metrics()

target_cols = [f"target_day_{d}" for d in range(1, 11)]

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")
loc_options = districts["loc"].tolist()
selected_loc = st.sidebar.selectbox("District", loc_options, index=0)

day_min, day_max = int(predictions["day_index"].min()), int(predictions["day_index"].max())
selected_day = st.sidebar.slider(
    "Test-period day (day_index)", min_value=day_min, max_value=day_max,
    value=day_min, help="0 = first day of the masked 2015-2025 test window."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**WBT danger thresholds**\n\n"
    "- Below 28°C: Safe\n"
    "- 28–32°C: Caution\n"
    "- 32–35°C: Danger\n"
    "- Above 35°C: Fatal (unsurvivable beyond ~6h)"
)

# ---------------- Region risk map (all districts, selected day) ----------------
st.subheader(f"Region-wide heat-risk snapshot — day_index {selected_day}")
snapshot = predictions[predictions["day_index"] == selected_day].copy()

if snapshot.empty:
    st.warning("No data for this day_index.")
else:
    fig_map = go.Figure()
    for level in RISK_ORDER:
        sub = snapshot[snapshot["risk_level"] == level]
        fig_map.add_trace(go.Scatter(
            x=sub["rel_lon"], y=sub["rel_lat"], mode="markers",
            name=level,
            marker=dict(size=14, color=RISK_COLORS[level], line=dict(width=1, color="white")),
            text=sub["loc"] + "<br>Max 10-day WBT: " + sub["max_forecast_wbt"].round(1).astype(str) + "°C",
            hoverinfo="text",
        ))
    fig_map.update_layout(
        xaxis_title="rel_lon", yaxis_title="rel_lat",
        height=420, legend_title="Risk level",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    counts = snapshot["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    cols = st.columns(4)
    for c, level in zip(cols, RISK_ORDER):
        c.metric(f"{level} districts", int(counts[level]))

st.markdown("---")

# ---------------- District detail ----------------
st.subheader(f"10-day forecast — {selected_loc}")
row = predictions[(predictions["loc"] == selected_loc) & (predictions["day_index"] == selected_day)]

if row.empty:
    st.warning("No forecast for this district/day combination.")
else:
    row = row.iloc[0]
    forecast_vals = [row[c] for c in target_cols]
    days = list(range(1, 11))

    fig = go.Figure()
    fig.add_hrect(y0=35, y1=45, fillcolor=RISK_COLORS["Fatal"], opacity=0.12, line_width=0)
    fig.add_hrect(y0=32, y1=35, fillcolor=RISK_COLORS["Danger"], opacity=0.12, line_width=0)
    fig.add_hrect(y0=28, y1=32, fillcolor=RISK_COLORS["Caution"], opacity=0.12, line_width=0)
    fig.add_hrect(y0=15, y1=28, fillcolor=RISK_COLORS["Safe"], opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(
        x=days, y=forecast_vals, mode="lines+markers", name="Forecast WBT",
        line=dict(color="#1565C0", width=3), marker=dict(size=9),
    ))
    fig.update_layout(
        xaxis_title="Days ahead", yaxis_title="Wet-Bulb Temperature (°C)",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[min(15, min(forecast_vals) - 2), max(38, max(forecast_vals) + 2)]),
    )
    st.plotly_chart(fig, use_container_width=True)

    peak_idx = int(np.argmax(forecast_vals))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak forecast WBT", f"{max(forecast_vals):.1f}°C", f"Day {peak_idx + 1}")
    m2.metric("Day-1 forecast", f"{forecast_vals[0]:.1f}°C")
    m3.metric("Day-10 forecast", f"{forecast_vals[-1]:.1f}°C")
    m4.markdown("**Overall risk**<br>" + risk_badge(row["risk_level"]), unsafe_allow_html=True)

st.markdown("---")

# ---------------- Model performance ----------------
with st.expander("\U0001F4CA Model performance & methodology", expanded=False):
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("10-day Weighted RMSE (validation)", f"{metrics['wrmse_10day']:.3f}°C")
        c2.metric("Same-day RMSE (validation)", f"{metrics['same_day_rmse']:.3f}°C")
        c3.metric("Features used", metrics["n_features"])

        per_day = metrics["per_day_rmse"]
        fig_perf = go.Figure(go.Bar(
            x=[f"Day {i+1}" for i in range(10)], y=per_day,
            marker_color="#1565C0", text=[f"{v:.2f}" for v in per_day], textposition="outside",
        ))
        fig_perf.update_layout(
            title="Validation RMSE by forecast horizon (weights decay 1.0 → 0.1)",
            yaxis_title="RMSE (°C)", height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        top_feats = metrics.get("top_features", {})
        if top_feats:
            feat_series = pd.Series(top_feats).sort_values()
            fig_imp = go.Figure(go.Bar(x=feat_series.values, y=feat_series.index, orientation="h",
                                        marker_color="#2E7D32"))
            fig_imp.update_layout(title="Top model features (LightGBM gain)", height=380,
                                   margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Run `py model/train_model.py` to generate validation metrics.")

    st.markdown(
        """
**Approach.** A LightGBM regressor predicts same-day WBT directly from NASA POWER
meteorological features (temperature, humidity, soil moisture, radiation, wind) plus
engineered lag/rolling statistics (1/3/7-day lags, 7-day rolling mean & std, deltas) computed
per district. Because the test period's meteorological drivers are themselves known
(NASA reanalysis, not a weather forecast), each of the next 10 days is scored directly and the
10-day-ahead target vector is read off from those same-day predictions — avoiding
compounding recursive-forecast error. `context.csv` seeds the first 30-day lag window per
district before the test period begins.

**Why WBT, not raw temperature.** WBT combines heat and humidity into a single physiological
heat-stress signal: above 32°C outdoor activity should stop, above 35°C conditions become
unsurvivable within hours even at rest.
"""
    )

st.caption("Prototype for Aether — IIIT Lucknow Climate Intelligence Challenge 2026.")
