"""
Trains a CPU-friendly LightGBM model to predict daily Wet-Bulb Temperature (WBT)
from NASA POWER meteorological features + engineered lag/rolling features.

This model is later applied day-by-day across the test period (2015-2025) to
produce the 10-day-ahead forecasts used by the Streamlit deployment (see app/app.py).

Run from the `ather hackathon` project root:
    py model/train_model.py
"""
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

ROOT = __file__.replace("model\\train_model.py", "").replace("model/train_model.py", "")

DAY_WEIGHTS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

MET_VARS = ["T2M", "T2M_MIN", "T2M_MAX", "RH2M", "QV2M", "TSOIL1", "TS",
            "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "WS10M", "PS",
            "GWETTOP", "PRECTOTCORR"]


def stull_wbt(T, RH):
    RH = np.clip(RH, 1, 100)
    return (T * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
            + np.arctan(T + RH)
            - np.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
            - 4.686035)


def add_base_features(df):
    df["temp_range"] = df["T2M_MAX"] - df["T2M_MIN"]
    df["temp_moisture"] = df["T2M"] * df["QV2M"]
    df["min_temp_moisture"] = df["T2M_MIN"] * df["QV2M"]
    df["heat_humidity"] = df["T2M"] * df["RH2M"]
    df["soil_moisture_avg"] = (df["GWETTOP"] + df["GWETROOT"]) / 2
    df["wet_soil_heat"] = df["TSOIL1"] * df["soil_moisture_avg"]
    df["solar_gap"] = df["CLRSKY_SFC_SW_DWN"] - df["ALLSKY_SFC_SW_DWN"]
    df["ts_diff"] = df["TS"] - df["T2M"]
    df["wd_sin"] = np.sin(np.deg2rad(df["WD10M"]))
    df["wd_cos"] = np.cos(np.deg2rad(df["WD10M"]))
    df["rh_squared"] = df["RH2M"] ** 2
    df["temp_squared"] = df["T2M"] ** 2
    df["vapor_pressure"] = (df["RH2M"] / 100) * (
        6.112 * np.exp(17.67 * df["T2M"] / (df["T2M"] + 243.5)))
    df["dewpoint"] = df["T2M"] - ((100 - df["RH2M"]) / 5)
    df["t_minus_dewpoint"] = df["T2M"] - df["dewpoint"]
    df["WBT_stull"] = stull_wbt(df["T2M"], df["RH2M"])
    return df


def add_lag_features(df, group_cols):
    for var in MET_VARS:
        g = df.groupby(group_cols)[var]
        df[f"{var}_lag1"] = g.shift(1)
        df[f"{var}_lag3"] = g.shift(3)
        df[f"{var}_lag7"] = g.shift(7)
        df[f"{var}_roll7_mean"] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        df[f"{var}_roll7_std"] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
        df[f"{var}_diff1"] = g.diff(1)

    if "WBT" in df.columns:
        g = df.groupby(group_cols)["WBT"]
        df["WBT_lag1"] = g.shift(1)
        df["WBT_lag3"] = g.shift(3)
        df["WBT_lag7"] = g.shift(7)
        df["WBT_roll7_mean"] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        df["WBT_diff1"] = g.diff(1)
    return df


BASE_FEATURES = [
    "rel_lat", "rel_lon",
    "T2M_MAX", "T2M_MIN", "T2M", "RH2M", "QV2M", "TSOIL1", "TS",
    "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "CLOUD_AMT",
    "GWETTOP", "GWETROOT", "WS10M", "PS", "PRECTOTCORR", "EVLAND",
    "temp_range", "temp_moisture", "min_temp_moisture", "heat_humidity",
    "soil_moisture_avg", "wet_soil_heat", "solar_gap", "ts_diff",
    "wd_sin", "wd_cos", "rh_squared", "temp_squared",
    "vapor_pressure", "dewpoint", "t_minus_dewpoint", "WBT_stull",
]
LAG_FEATURES = []
for v in MET_VARS:
    LAG_FEATURES += [f"{v}_lag1", f"{v}_lag3", f"{v}_lag7",
                      f"{v}_roll7_mean", f"{v}_roll7_std", f"{v}_diff1"]
FEATURE_COLS = BASE_FEATURES + LAG_FEATURES


def main():
    t0 = time.time()
    print("Loading data/train.csv ...")
    train = pd.read_csv("data/train.csv")
    train["date"] = pd.to_datetime(train["date"])
    train = train.sort_values(["rel_lat", "rel_lon", "date"]).reset_index(drop=True)

    print("Engineering features ...")
    train = add_base_features(train)
    train = add_lag_features(train, ["rel_lat", "rel_lon"])
    train = train.dropna(subset=["T2M_lag7"]).reset_index(drop=True)

    feature_cols = [c for c in FEATURE_COLS if c in train.columns]
    print(f"Using {len(feature_cols)} features on {len(train):,} rows")

    split_date = train["date"].quantile(0.85)
    tr = train[train["date"] <= split_date]
    vl = train[train["date"] > split_date]
    print(f"Train: {len(tr):,}  Valid: {len(vl):,}")

    tr_X, tr_y = tr[feature_cols].values, tr["WBT"].values
    vl_X, vl_y = vl[feature_cols].values, vl["WBT"].values

    model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )

    print("Training LightGBM (CPU) ...")
    model.fit(
        tr_X, tr_y,
        eval_set=[(vl_X, vl_y)],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )

    vl_pred = model.predict(vl_X)
    same_day_rmse = float(np.sqrt(mean_squared_error(vl_y, vl_pred)))
    same_day_mae = float(mean_absolute_error(vl_y, vl_pred))
    same_day_r2 = float(r2_score(vl_y, vl_pred))
    print(f"\nSame-day RMSE: {same_day_rmse:.4f}  MAE: {same_day_mae:.4f}  R2: {same_day_r2:.4f}")

    # Simulate the 10-day-ahead WRMSE the same way the real submission does:
    # look up the model's same-day prediction on future days (features are
    # known for every day in the dataset, so no recursive forecasting is
    # needed -- see README for why).
    vl = vl.copy()
    vl["WBT_pred"] = vl_pred
    for d in range(1, 11):
        vl[f"true_{d}"] = vl.groupby(["rel_lat", "rel_lon"])["WBT"].shift(-d)
        vl[f"pred_{d}"] = vl.groupby(["rel_lat", "rel_lon"])["WBT_pred"].shift(-d)

    vl_clean = vl.dropna(subset=[f"true_{d}" for d in range(1, 11)] + [f"pred_{d}" for d in range(1, 11)])
    y_true = np.column_stack([vl_clean[f"true_{d}"].values for d in range(1, 11)])
    y_pred = np.column_stack([vl_clean[f"pred_{d}"].values for d in range(1, 11)])
    wrmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2 * DAY_WEIGHTS)))
    per_day_rmse = [float(np.sqrt(np.mean((y_true[:, d] - y_pred[:, d]) ** 2))) for d in range(10)]
    print(f"10-day weighted WRMSE: {wrmse:.4f}")

    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 15 features:")
    print(feat_imp.head(15))

    joblib.dump(model, "model/wbt_lgbm.joblib")
    with open("model/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    metrics = {
        "same_day_rmse": same_day_rmse,
        "same_day_mae": same_day_mae,
        "same_day_r2": same_day_r2,
        "wrmse_10day": wrmse,
        "per_day_rmse": per_day_rmse,
        "day_weights": DAY_WEIGHTS.tolist(),
        "n_train_rows": int(len(tr)),
        "n_valid_rows": int(len(vl)),
        "n_features": len(feature_cols),
        "n_boosted_rounds": int(model.booster_.num_trees()),
        "top_features": feat_imp.head(15).to_dict(),
        "train_seconds": time.time() - t0,
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model + metrics. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
