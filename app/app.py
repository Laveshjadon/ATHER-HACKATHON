"""
Streamlit prototype for the IIIT Lucknow Climate Intelligence Challenge 2026
(Aether AI/ML Club) -- 10-day Wet-Bulb Temperature (WBT) heat-risk forecaster.

Run locally:
    streamlit run app/app.py

Deploy: push this repo to GitHub and point Streamlit Community Cloud at
app/app.py (see README.md).
"""
import io
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
ROOT_DIR = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")
sys.path.insert(0, MODEL_DIR)
from train_model import add_base_features, add_lag_features  # noqa: E402

DAY_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
RISK_COLORS = {"Safe": "#2E7D32", "Caution": "#F9A825", "Danger": "#EF6C00", "Fatal": "#C62828"}
RISK_ORDER = ["Safe", "Caution", "Danger", "Fatal"]

RAW_INPUT_COLS = [
    "date", "rel_lat", "rel_lon",
    "T2M_MAX", "T2M_MIN", "T2M", "RH2M", "QV2M", "TSOIL1",
    "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "CLOUD_AMT",
    "GWETTOP", "GWETROOT", "WS10M", "WD10M", "PS", "PRECTOTCORR", "TS", "EVLAND",
]

st.set_page_config(page_title="UP Heat-Risk Forecaster", page_icon="\U0001F321️", layout="wide")


@st.cache_data
def load_predictions():
    return pd.read_parquet(os.path.join(DATA_DIR, "predictions.parquet"))


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


@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_DIR, "wbt_lgbm.joblib")
    feat_path = os.path.join(MODEL_DIR, "feature_cols.json")
    if not (os.path.exists(model_path) and os.path.exists(feat_path)):
        return None, None
    model = joblib.load(model_path)
    with open(feat_path) as f:
        feature_cols = json.load(f)
    return model, feature_cols


def risk_badge(level):
    color = RISK_COLORS.get(level, "#999")
    return f'<span style="background:{color};color:white;padding:3px 12px;border-radius:12px;font-weight:600;">{level}</span>'


def risk_level_for(w):
    if w >= 35:
        return "Fatal"
    if w >= 32:
        return "Danger"
    if w >= 28:
        return "Caution"
    return "Safe"


def forecast_chart(days, forecast_vals, title=None):
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
        title=title, xaxis_title="Days ahead", yaxis_title="Wet-Bulb Temperature (°C)",
        height=380, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        yaxis=dict(range=[min(15, min(forecast_vals) - 2), max(38, max(forecast_vals) + 2)]),
    )
    return fig


def sample_template_csv():
    rows = []
    base = {
        "T2M_MAX": 42.0, "T2M_MIN": 29.0, "T2M": 36.0, "RH2M": 45.0, "QV2M": 16.5,
        "TSOIL1": 33.0, "ALLSKY_SFC_SW_DWN": 24.0, "CLRSKY_SFC_SW_DWN": 26.5, "CLOUD_AMT": 20.0,
        "GWETTOP": 0.35, "GWETROOT": 0.4, "WS10M": 2.8, "WD10M": 210.0, "PS": 98.5,
        "PRECTOTCORR": 0.0, "TS": 37.5, "EVLAND": 1.2,
    }
    for i in range(30):
        row = {"date": (pd.Timestamp("2025-05-01") + pd.Timedelta(days=i)).date().isoformat(),
               "rel_lat": 12.4, "rel_lon": 5.1}
        row.update({k: v + np.sin(i / 3) * 1.5 for k, v in base.items()})
        rows.append(row)
    return pd.DataFrame(rows)[RAW_INPUT_COLS].to_csv(index=False)


def run_batch_inference(raw_df, model, feature_cols):
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["rel_lat", "rel_lon", "date"]).reset_index(drop=True)

    extended_parts = []
    for (lat, lon), g in df.groupby(["rel_lat", "rel_lon"]):
        g = g.sort_values("date").reset_index(drop=True)
        last_row = g.iloc[-1]
        last_date = g["date"].max()
        future_rows = []
        for d in range(1, 11):
            fr = last_row.copy()
            fr["date"] = last_date + pd.Timedelta(days=d)
            future_rows.append(fr)
        g_ext = pd.concat([g, pd.DataFrame(future_rows)], ignore_index=True)
        extended_parts.append(g_ext)
    extended = pd.concat(extended_parts, ignore_index=True)

    extended = add_base_features(extended)
    extended = add_lag_features(extended, ["rel_lat", "rel_lon"])
    for col in feature_cols:
        if col not in extended.columns:
            extended[col] = 0.0
        extended[col] = extended[col].fillna(0)

    extended["WBT_pred"] = model.predict(extended[feature_cols].values)

    results = []
    for (lat, lon), g in extended.groupby(["rel_lat", "rel_lon"]):
        g = g.sort_values("date").reset_index(drop=True)
        n_hist = len(g) - 10
        forecast_vals = g["WBT_pred"].iloc[n_hist:n_hist + 10].tolist()
        results.append({"rel_lat": lat, "rel_lon": lon, "forecast": forecast_vals})
    return results


st.title("\U0001F321️ Uttar Pradesh Heat-Risk Forecaster")
st.caption(
    "10-day-ahead Wet-Bulb Temperature (WBT) forecasts across 125 districts of UP & neighboring "
    "regions — built for the IIIT Lucknow Climate Intelligence Challenge 2026 (Aether)."
)

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

tab_dashboard, tab_batch, tab_perf = st.tabs(
    ["\U0001F5FA️ Regional Dashboard", "\U0001F52E Batch Inference (New Data)", "\U0001F4CA Model Performance"]
)

# ============================================================
# TAB 1 — Regional dashboard (precomputed test-period forecasts)
# ============================================================
with tab_dashboard:
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
    st.subheader(f"10-day forecast — {selected_loc}")
    row = predictions[(predictions["loc"] == selected_loc) & (predictions["day_index"] == selected_day)]

    if row.empty:
        st.warning("No forecast for this district/day combination.")
    else:
        row = row.iloc[0]
        forecast_vals = [row[c] for c in target_cols]
        days = list(range(1, 11))
        st.plotly_chart(forecast_chart(days, forecast_vals), use_container_width=True)

        peak_idx = int(np.argmax(forecast_vals))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak forecast WBT", f"{max(forecast_vals):.1f}°C", f"Day {peak_idx + 1}")
        m2.metric("Day-1 forecast", f"{forecast_vals[0]:.1f}°C")
        m3.metric("Day-10 forecast", f"{forecast_vals[-1]:.1f}°C")
        m4.markdown("**Overall risk**<br>" + risk_badge(row["risk_level"]), unsafe_allow_html=True)

# ============================================================
# TAB 2 — Batch inference on user-uploaded data
# ============================================================
with tab_batch:
    st.subheader("Run the model on your own data")
    st.markdown(
        "Upload a CSV of daily meteorological features (same schema as `data/test.csv`) for one "
        "or more districts — at least the last **30 days** per district, ending on the day you "
        "want to forecast from."
    )
    st.info(
        "**Assumption:** because the uploaded data has no known future days, the next 10 days' "
        "meteorological drivers are persisted from the most recent uploaded day (a standard "
        "no-external-forecast baseline) — the model then predicts WBT from that persisted "
        "state plus your uploaded history's lag/rolling features. Forecast quality depends on "
        "how representative the last uploaded day is of the days ahead."
    )

    st.download_button(
        "Download sample input template (CSV)",
        data=sample_template_csv(),
        file_name="batch_inference_template.csv",
        mime="text/csv",
    )

    model, feature_cols = load_model()
    if model is None:
        st.warning(
            "No trained model found at `model/wbt_lgbm.joblib`. Run `py model/train_model.py` "
            "from the project root first."
        )
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
                missing = [c for c in RAW_INPUT_COLS if c not in raw_df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    with st.spinner("Running inference..."):
                        results = run_batch_inference(raw_df, model, feature_cols)

                    st.success(f"Scored {len(results)} district(s).")
                    days = list(range(1, 11))
                    out_rows = []
                    for r in results:
                        label = f"({r['rel_lat']:.2f}, {r['rel_lon']:.2f})"
                        st.plotly_chart(
                            forecast_chart(days, r["forecast"], title=f"10-day forecast — {label}"),
                            use_container_width=True,
                        )
                        out_row = {"rel_lat": r["rel_lat"], "rel_lon": r["rel_lon"]}
                        out_row.update({f"target_day_{i+1}": v for i, v in enumerate(r["forecast"])})
                        out_rows.append(out_row)

                    out_df = pd.DataFrame(out_rows)
                    buf = io.StringIO()
                    out_df.to_csv(buf, index=False)
                    st.download_button(
                        "Download forecast results (CSV)", data=buf.getvalue(),
                        file_name="batch_inference_results.csv", mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Could not process file: {e}")

# ============================================================
# TAB 3 — Model performance
# ============================================================
with tab_perf:
    st.subheader("Validation performance")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("10-day Weighted RMSE", f"{metrics['wrmse_10day']:.3f}°C")
        c2.metric("Same-day RMSE", f"{metrics['same_day_rmse']:.3f}°C")
        c3.metric("Same-day MAE", f"{metrics.get('same_day_mae', float('nan')):.3f}°C")
        c4.metric("Same-day R²", f"{metrics.get('same_day_r2', float('nan')):.4f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Training rows", f"{metrics['n_train_rows']:,}")
        c6.metric("Validation rows", f"{metrics['n_valid_rows']:,}")
        c7.metric("Features used", metrics["n_features"])

        per_day = metrics["per_day_rmse"]
        fig_perf = go.Figure(go.Bar(
            x=[f"Day {i+1}" for i in range(10)], y=per_day,
            marker_color="#1565C0", text=[f"{v:.2f}" for v in per_day], textposition="outside",
        ))
        fig_perf.update_layout(
            title="Validation RMSE by forecast horizon (competition weights decay 1.0 → 0.1)",
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
