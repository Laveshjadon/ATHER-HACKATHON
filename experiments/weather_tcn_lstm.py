import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
import copy
import gc

# Clear GPU memory
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = train_df.copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["rel_lat", "rel_lon", "date"]).reset_index(drop=True)

# ==========================================
# FEATURE ENGINEERING
# ==========================================
df["temp_range"] = df["T2M_MAX"] - df["T2M_MIN"]

df["temp_moisture"] = df["T2M"] * df["QV2M"]
df["min_temp_moisture"] = df["T2M_MIN"] * df["QV2M"]
df["heat_humidity"] = df["T2M"] * df["RH2M"]

df["soil_moisture_avg"] = (df["GWETTOP"] + df["GWETROOT"]) / 2
df["wet_soil_heat"] = df["TSOIL1"] * df["soil_moisture_avg"]

df["solar_gap"] = df["CLRSKY_SFC_SW_DWN"] - df["ALLSKY_SFC_SW_DWN"]

# Wind direction cyclical encoding
df["wd_sin"] = np.sin(np.deg2rad(df["WD10M"]))
df["wd_cos"] = np.cos(np.deg2rad(df["WD10M"]))

# Time features
df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month

df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# --- LAG FEATURES ---
for lag in [1, 2, 3, 7]:
    df[f"WBT_lag_{lag}"] = df.groupby(
        ["rel_lat", "rel_lon"])["WBT"].shift(lag)

# --- ROLLING FEATURES ---
df["WBT_roll3_mean"] = df.groupby(
    ["rel_lat", "rel_lon"])["WBT"].transform(
        lambda x: x.shift(1).rolling(3).mean())

df["WBT_roll7_mean"] = df.groupby(
    ["rel_lat", "rel_lon"])["WBT"].transform(
        lambda x: x.shift(1).rolling(7).mean())

df["WBT_roll7_std"] = df.groupby(
    ["rel_lat", "rel_lon"])["WBT"].transform(
        lambda x: x.shift(1).rolling(7).std())

# --- DIFF & EMA ---
df["WBT_diff_1"] = df.groupby(
    ["rel_lat", "rel_lon"])["WBT"].diff(1)

df["WBT_ema_7"] = df.groupby(
    ["rel_lat", "rel_lon"])["WBT"].transform(
        lambda x: x.shift(1).ewm(span=7, adjust=False).mean())

# --- TEMP LAGS ---
df["T2M_lag_1"] = df.groupby(
    ["rel_lat", "rel_lon"])["T2M"].shift(1)

df["temp_dewpoint_proxy"] = df["T2M"] - ((100 - df["RH2M"]) / 5)

# --- STULL FORMULA ---
def stull_wbt(T, RH):
    wbt = (T * np.arctan(0.151977 * (RH + 8.313659)**0.5)
           + np.arctan(T + RH)
           - np.arctan(RH - 1.676331)
           + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
           - 4.686035)
    return wbt

df["WBT_stull"] = stull_wbt(df["T2M"], df["RH2M"])
df["WBT_residual"] = df["WBT"] - df["WBT_stull"]

# --- HUMIDITY INTERACTION FEATURES ---
df["rh_temp_product"] = df["RH2M"] * df["T2M"]
df["rh_squared"] = df["RH2M"] ** 2
df["temp_squared"] = df["T2M"] ** 2
df["rh_log"] = np.log1p(df["RH2M"])
df["vapor_pressure"] = (df["RH2M"] / 100) * (
    6.112 * np.exp(17.67 * df["T2M"] / (df["T2M"] + 243.5))
)

# --- PSYCHROMETRIC FEATURES ---
df["dewpoint"] = df["T2M"] - ((100 - df["RH2M"]) / 5)
df["wet_bulb_approx"] = (
    df["T2M"] * np.arctan(0.151977 * (df["RH2M"] + 8.313659)**0.5)
    + np.arctan(df["T2M"] + df["RH2M"])
    - np.arctan(df["RH2M"] - 1.676331)
    + 0.00391838 * df["RH2M"]**1.5
    * np.arctan(0.023101 * df["RH2M"])
    - 4.686035
)
df["T_RH_ratio"] = df["T2M"] / (df["RH2M"] + 1)
df["T_minus_dewpoint"] = df["T2M"] - df["dewpoint"]
df["RH_deficit"] = 100 - df["RH2M"]

df = df.dropna().reset_index(drop=True)

target_col = "WBT"

feature_cols = [
    "rel_lat", "rel_lon",

    "T2M_MAX", "T2M_MIN", "T2M",
    "RH2M", "QV2M",
    "TSOIL1", "TS",

    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "CLOUD_AMT",

    "GWETTOP", "GWETROOT",
    "WS10M",

    "wd_sin", "wd_cos",

    "PS", "PRECTOTCORR",
    "EVLAND",

    "temp_range",
    "temp_moisture",
    "min_temp_moisture",
    "heat_humidity",
    "soil_moisture_avg",
    "wet_soil_heat",
    "solar_gap",

    "day_sin", "day_cos",
    "month_sin", "month_cos",

    "WBT_lag_1", "WBT_lag_2", "WBT_lag_3", "WBT_lag_7",
    "WBT_roll3_mean", "WBT_roll7_mean", "WBT_roll7_std",
    "WBT_diff_1", "WBT_ema_7",
    "T2M_lag_1",
    "temp_dewpoint_proxy",
    "WBT_stull",
    "rh_temp_product",
    "rh_squared",
    "temp_squared",
    "rh_log",
    "vapor_pressure",
    "dewpoint",
    "wet_bulb_approx",
    "T_RH_ratio",
    "T_minus_dewpoint",
    "RH_deficit",
]

# ==========================================
# TIME SPLIT
# ==========================================
split_date = df["date"].quantile(0.8)

train_data = df[df["date"] <= split_date].copy()
valid_data = df[df["date"] > split_date].copy()

# Free original df
del df
gc.collect()

print("Train data:", train_data.shape)
print("Valid data:", valid_data.shape)


# --- Save original values BEFORE scaling (for Stull + LightGBM later) ---
train_data["T2M_original"] = train_data["T2M"].copy()
train_data["RH2M_original"] = train_data["RH2M"].copy()
train_data["WBT_original"] = train_data["WBT"].copy()
valid_data["T2M_original"] = valid_data["T2M"].copy()
valid_data["RH2M_original"] = valid_data["RH2M"].copy()
valid_data["WBT_original"] = valid_data["WBT"].copy()

# ==========================================
# SCALING
# ==========================================
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

train_data[feature_cols] = feature_scaler.fit_transform(train_data[feature_cols])
valid_data[feature_cols] = feature_scaler.transform(valid_data[feature_cols])

train_data[[target_col]] = target_scaler.fit_transform(train_data[[target_col]])
valid_data[[target_col]] = target_scaler.transform(valid_data[[target_col]])

# ==========================================
# LIGHTGBM DIRECT WBT MODEL
# ==========================================
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

lgb_feature_cols = feature_cols

lgb_train_X = train_data[lgb_feature_cols].values
lgb_train_y = train_data["WBT_original"].values

lgb_valid_X = valid_data[lgb_feature_cols].values
lgb_valid_y = valid_data["WBT_original"].values

# Save for final eval
valid_T2M_original = valid_data["T2M_original"].values.copy()
valid_RH2M_original = valid_data["RH2M_original"].values.copy()
valid_WBT_original = valid_data["WBT_original"].values.copy()

lgb_model = lgb.LGBMRegressor(
    n_estimators=20000,
    learning_rate=0.05,
    max_depth=12,
    num_leaves=511,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    device='gpu'
)

# Compute sample weights (WRMSE-aligned)
sample_weights = np.abs(lgb_train_y) / np.sum(np.abs(lgb_train_y))
sample_weights = sample_weights * len(lgb_train_y)

lgb_model.fit(
    lgb_train_X, lgb_train_y,
    sample_weight=sample_weights,
    eval_set=[(lgb_valid_X, lgb_valid_y)],
    callbacks=[lgb.early_stopping(300), lgb.log_evaluation(100)]
)

# Predict WBT directly
final_preds = lgb_model.predict(lgb_valid_X)
final_actual = valid_data["WBT_original"].values

# Evaluate
final_errors = final_preds - final_actual
final_weights = np.abs(final_actual) / np.sum(np.abs(final_actual))
final_wrmse = np.sqrt(np.sum(final_weights * final_errors**2))
final_rmse = np.sqrt(np.mean(final_errors**2))
final_mae = np.mean(np.abs(final_errors))

print("\n=== LIGHTGBM DIRECT WBT ===")
print(f"WRMSE : {final_wrmse:.5f}")
print(f"RMSE  : {final_rmse:.5f}")
print(f"MAE   : {final_mae:.5f}")

# --- Feature Importances ---
feat_imp = pd.Series(
    lgb_model.feature_importances_,
    index=lgb_feature_cols
).sort_values(ascending=False)
print("\n=== TOP 20 FEATURE IMPORTANCES ===")
print(feat_imp.head(20))

# --- Save LightGBM predictions to CSV ---
pred_df = pd.DataFrame({
    "predicted_WBT": final_preds,
    "actual_WBT": final_actual
})
pred_df.to_csv("lgb_valid_predictions.csv", index=False)
print("\nSaved LightGBM predictions to lgb_valid_predictions.csv")

# ==========================================
# CATBOOST MODEL
# ==========================================
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=10000,
    learning_rate=0.05,
    depth=10,
    l2_leaf_reg=3,
    subsample=0.8,
    colsample_bylevel=0.8,
    min_data_in_leaf=20,
    eval_metric='RMSE',
    early_stopping_rounds=300,
    random_seed=42,
    task_type='GPU',
    verbose=200
)

cat_model.fit(
    lgb_train_X, lgb_train_y,
    eval_set=(lgb_valid_X, lgb_valid_y),
    sample_weight=sample_weights
)

cat_preds = cat_model.predict(lgb_valid_X)

# CatBoost standalone eval
cat_errors = cat_preds - final_actual
cat_weights = np.abs(final_actual) / np.sum(np.abs(final_actual))
cat_wrmse = np.sqrt(np.sum(cat_weights * cat_errors**2))
cat_rmse = np.sqrt(np.mean(cat_errors**2))
cat_mae = np.mean(np.abs(cat_errors))

print("\n=== CATBOOST DIRECT WBT ===")
print(f"WRMSE : {cat_wrmse:.5f}")
print(f"RMSE  : {cat_rmse:.5f}")
print(f"MAE   : {cat_mae:.5f}")

# ==========================================
# ENSEMBLE (LightGBM + CatBoost)
# ==========================================
lgb_preds = lgb_model.predict(lgb_valid_X)

print("\n=== ENSEMBLE BLENDS ===")
best_wrmse = float("inf")
best_blend = None
best_w = None

for w in [0.4, 0.5, 0.6]:
    blend = w * lgb_preds + (1 - w) * cat_preds
    errors = blend - final_actual
    weights = np.abs(final_actual) / np.sum(np.abs(final_actual))
    wrmse = np.sqrt(np.sum(weights * errors**2))
    print(f"LGB {w:.1f} | Cat {1-w:.1f} | WRMSE: {wrmse:.5f}")

    if wrmse < best_wrmse:
        best_wrmse = wrmse
        best_blend = blend
        best_w = w

print(f"\nBest blend: LGB={best_w:.1f}, CAT={1-best_w:.1f} | WRMSE: {best_wrmse:.5f}")

# --- Save best ensemble predictions ---
ensemble_df = pd.DataFrame({
    "predicted_WBT": best_blend,
    "actual_WBT": final_actual
})
ensemble_df.to_csv("ensemble_predictions.csv", index=False)
print("Saved ensemble predictions to ensemble_predictions.csv")

# Free arrays
del lgb_train_X, lgb_train_y, lgb_valid_X, lgb_valid_y
del train_data, valid_data
gc.collect()

print("\nDone.")
