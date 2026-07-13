import pandas as pd
import numpy as np
import gc
import os

# ==========================================
# STEP 1: LOAD FILES
# ==========================================
def load_csv(name):
    paths = [name, f"/kaggle/input/{name}", f"../input/{name}"]
    kaggle_base = "/kaggle/input"
    if os.path.exists(kaggle_base):
        for folder in os.listdir(kaggle_base):
            paths.append(os.path.join(kaggle_base, folder, name))
    for p in paths:
        if os.path.exists(p):
            print(f"Loading {name} from: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find {name}")

train_df = load_csv("train.csv")
test_df = load_csv("test.csv")
context_df = load_csv("context.csv")
sample_sub = load_csv("sample_submission.csv")

print("\nTrain:", train_df.shape)
print("Test:", test_df.shape)
print("Context:", context_df.shape)
print("Sample Sub:", sample_sub.shape)

# ==========================================
# STEP 2: STULL + FEATURE ENGINEERING
# ==========================================
def stull_wbt(T, RH):
    return (T * np.arctan(0.151977 * (RH + 8.313659)**0.5)
            + np.arctan(T + RH)
            - np.arctan(RH - 1.676331)
            + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
            - 4.686035)

# Key meteorological variables to create lags for
met_vars = ["T2M", "T2M_MIN", "QV2M", "TSOIL1", "TS",
            "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "WS10M", "PS",
            "GWETTOP", "PRECTOTCORR"]

def add_base_features(df):
    """Static features (no temporal context needed)."""
    df["temp_range"] = df["T2M"] - df["T2M_MIN"]
    df["temp_squared"] = df["T2M"] ** 2
    df["temp_x_qv"] = df["T2M"] * df["QV2M"]
    df["qv_squared"] = df["QV2M"] ** 2
    df["qv_log"] = np.log1p(df["QV2M"])
    
    # Derive RH from QV2M
    es = 6.112 * np.exp(17.67 * df["T2M"] / (df["T2M"] + 243.5))
    qs = 0.622 * es / (df["PS"] * 10 - es)
    df["RH_derived"] = np.clip((df["QV2M"] / 1000.0) / qs * 100.0, 0, 100)
    df["T_x_RH"] = df["T2M"] * df["RH_derived"]
    
    # Vapor pressure
    df["actual_vp"] = df["QV2M"] * df["PS"] * 10 / (0.622 + df["QV2M"] / 1000.0)
    df["sat_vp"] = es
    df["vp_deficit"] = es - df["actual_vp"]
    
    # Dewpoint
    vp_safe = np.clip(df["actual_vp"], 0.001, None)
    df["dewpoint"] = (243.5 * np.log(vp_safe / 6.112)) / (17.67 - np.log(vp_safe / 6.112))
    df["t_minus_dp"] = df["T2M"] - df["dewpoint"]
    
    # Stull with derived RH
    df["WBT_stull_t2m"] = stull_wbt(df["T2M"], df["RH_derived"])
    
    # Soil & radiation
    df["soil_moisture_avg"] = (df["GWETTOP"] + df["GWETROOT"]) / 2
    df["solar_gap"] = df["CLRSKY_SFC_SW_DWN"] - df["ALLSKY_SFC_SW_DWN"]
    df["wet_soil_heat"] = df["TSOIL1"] * df["soil_moisture_avg"]
    df["ts_diff"] = df["TS"] - df["T2M"]
    
    # Wind
    df["wd_sin"] = np.sin(np.deg2rad(df["WD10M"]))
    df["wd_cos"] = np.cos(np.deg2rad(df["WD10M"]))
    
    return df

def add_lag_features(df, group_cols):
    """Add rolling window features per location."""
    for var in met_vars:
        if var not in df.columns:
            continue
        g = df.groupby(group_cols)[var]
        
        # Lags 1, 3, 7
        df[f"{var}_lag1"] = g.shift(1)
        df[f"{var}_lag3"] = g.shift(3)
        df[f"{var}_lag7"] = g.shift(7)
        
        # Rolling stats (7-day)
        df[f"{var}_roll7_mean"] = g.transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        df[f"{var}_roll7_std"] = g.transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).std())
        
        # Diff
        df[f"{var}_diff1"] = g.diff(1)
    
    # WBT lags (only for train/context where WBT exists)
    if "WBT" in df.columns:
        g = df.groupby(group_cols)["WBT"]
        df["WBT_lag1"] = g.shift(1)
        df["WBT_lag3"] = g.shift(3)
        df["WBT_lag7"] = g.shift(7)
        df["WBT_roll7_mean"] = g.transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        df["WBT_diff1"] = g.diff(1)
    
    return df

# Build feature column list
base_features = [
    "rel_lat", "rel_lon",
    "T2M_MIN", "T2M", "QV2M", "TSOIL1", "TS",
    "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "CLOUD_AMT",
    "GWETTOP", "GWETROOT", "WS10M", "PS", "PRECTOTCORR", "EVLAND",
    "temp_range", "temp_squared", "temp_x_qv",
    "qv_squared", "qv_log",
    "RH_derived", "T_x_RH",
    "actual_vp", "sat_vp", "vp_deficit",
    "dewpoint", "t_minus_dp", "WBT_stull_t2m",
    "soil_moisture_avg", "solar_gap", "wet_soil_heat", "ts_diff",
    "wd_sin", "wd_cos",
]

lag_features = []
for var in met_vars:
    lag_features += [f"{var}_lag1", f"{var}_lag3", f"{var}_lag7",
                     f"{var}_roll7_mean", f"{var}_roll7_std", f"{var}_diff1"]

feature_cols = base_features + lag_features

# ==========================================
# STEP 3: PREPARE TRAIN DATA WITH LAGS
# ==========================================
print("\n=== PREPARING TRAIN WITH TEMPORAL FEATURES ===")

train_df["date"] = pd.to_datetime(train_df["date"])
train_df = train_df.sort_values(["rel_lat", "rel_lon", "date"]).reset_index(drop=True)
train_df = add_base_features(train_df)
train_df = add_lag_features(train_df, ["rel_lat", "rel_lon"])

# Drop rows without enough lag history (first ~7 per location)
train_df = train_df.dropna(subset=[f"T2M_lag7"]).reset_index(drop=True)
print(f"Train after lags: {train_df.shape}")

# Split
split_date = train_df["date"].quantile(0.8)
tr = train_df[train_df["date"] <= split_date]
vl = train_df[train_df["date"] > split_date]
print(f"Train: {tr.shape}, Valid: {vl.shape}")

del train_df
gc.collect()

# ==========================================
# STEP 4: TRAIN LIGHTGBM
# ==========================================
print("\n=== TRAINING LIGHTGBM (GPU) ===")
import lightgbm as lgb

# Filter feature_cols to only existing columns
feature_cols = [c for c in feature_cols if c in tr.columns]
print(f"Using {len(feature_cols)} features")

tr_X = tr[feature_cols].values
tr_y = tr["WBT"].values
vl_X = vl[feature_cols].values
vl_y = vl["WBT"].values

sw = np.abs(tr_y) / np.sum(np.abs(tr_y)) * len(tr_y)

model = lgb.LGBMRegressor(
    n_estimators=20000,
    learning_rate=0.01,
    max_depth=12,
    num_leaves=511,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    min_child_samples=20,
    reg_alpha=0.05,
    reg_lambda=0.05,
    random_state=42,
    n_jobs=-1,
    device='gpu',
)

model.fit(
    tr_X, tr_y,
    sample_weight=sw,
    eval_set=[(vl_X, vl_y)],
    callbacks=[lgb.early_stopping(300), lgb.log_evaluation(500)]
)

vl_pred = model.predict(vl_X)
same_day_rmse = np.sqrt(np.mean((vl_pred - vl_y)**2))
print(f"\nSame-day RMSE: {same_day_rmse:.5f}")

feat_imp = pd.Series(model.feature_importances_, index=feature_cols)
print("\nTop 20 features:")
print(feat_imp.sort_values(ascending=False).head(20))

del tr_X, sw
gc.collect()

# ==========================================
# STEP 5: LOCAL WRMSE
# ==========================================
print("\n=== LOCAL WRMSE ===")
day_weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

vl = vl.copy()
vl["WBT_pred"] = model.predict(vl[feature_cols].values)

for d in range(1, 11):
    vl[f"true_{d}"] = vl.groupby(["rel_lat", "rel_lon"])["WBT"].shift(-d)
    vl[f"pred_{d}"] = vl.groupby(["rel_lat", "rel_lon"])["WBT_pred"].shift(-d)

vl_clean = vl.dropna(subset=[f"true_{d}" for d in range(1, 11)] +
                             [f"pred_{d}" for d in range(1, 11)])

y_true = np.column_stack([vl_clean[f"true_{d}"].values for d in range(1, 11)])
y_pred = np.column_stack([vl_clean[f"pred_{d}"].values for d in range(1, 11)])
wrmse = np.sqrt(np.mean((y_true - y_pred)**2 * day_weights))
print(f"Lookup WRMSE: {wrmse:.5f}")

for d in range(1, 11):
    rmse = np.sqrt(np.mean((y_true[:, d-1] - y_pred[:, d-1])**2))
    print(f"  Day {d:2d} (w={day_weights[d-1]:.1f}): RMSE = {rmse:.5f}")

del vl, vl_X, vl_y
gc.collect()

# ==========================================
# STEP 6: PREPARE TEST WITH CONTEXT LAGS
# ==========================================
print("\n=== PREPARING TEST DATA ===")

# Merge context + test per location for lag features
context_df["date"] = pd.to_datetime(context_df["date"])
context_df = context_df.sort_values(["rel_lat", "rel_lon", "date"]).reset_index(drop=True)

# Give context a day_index before test starts
# Context has 30 days per location, test starts at day_index=0
# So context days = -30, -29, ..., -1
context_df["day_index"] = context_df.groupby(
    ["rel_lat", "rel_lon"]).cumcount() - 30

# Combine
test_df = test_df.sort_values(["rel_lat", "rel_lon", "day_index"]).reset_index(drop=True)

# Keep only needed columns
keep_cols = ["row_id", "rel_lat", "rel_lon", "day_index"] + [
    c for c in test_df.columns if c not in ["row_id", "rel_lat", "rel_lon", "day_index"]
]
ctx_keep = ["row_id", "rel_lat", "rel_lon", "day_index", "WBT"] + [
    c for c in context_df.columns 
    if c not in ["row_id", "rel_lat", "rel_lon", "day_index", "date", "WBT"]
]

combined = pd.concat([
    context_df[ctx_keep],
    test_df[keep_cols]
], ignore_index=True)

combined = combined.sort_values(["rel_lat", "rel_lon", "day_index"]).reset_index(drop=True)
print(f"Combined (context+test): {combined.shape}")

# Add features
combined = add_base_features(combined)
combined = add_lag_features(combined, ["rel_lat", "rel_lon"])

# Keep only test rows (day_index >= 0)
test_feat = combined[combined["day_index"] >= 0].copy()
test_feat = test_feat.reset_index(drop=True)
print(f"Test with features: {test_feat.shape}")

# Fill any remaining NaN in lag features with 0
for col in lag_features:
    if col in test_feat.columns:
        test_feat[col] = test_feat[col].fillna(0)

del combined
gc.collect()

# ==========================================
# STEP 7: PREDICT & BUILD SUBMISSION
# ==========================================
print("\n=== BUILDING SUBMISSION ===")

# Predict WBT for every test row
avail_features = [c for c in feature_cols if c in test_feat.columns]
test_feat["WBT_pred"] = model.predict(test_feat[avail_features].values)
print(f"WBT_pred range: [{test_feat['WBT_pred'].min():.2f}, {test_feat['WBT_pred'].max():.2f}]")

# Shift for 10-day targets (lookup future predictions)
target_cols = [f"target_day_{d}" for d in range(1, 11)]

for d in range(1, 11):
    test_feat[f"target_day_{d}"] = test_feat.groupby(
        ["rel_lat", "rel_lon"]
    )["WBT_pred"].shift(-d)

for col in target_cols:
    test_feat[col] = test_feat.groupby(["rel_lat", "rel_lon"])[col].ffill()
    test_feat[col] = test_feat.groupby(["rel_lat", "rel_lon"])[col].bfill()

# Build submission
sub = sample_sub[["row_id"]].copy()
sub = sub.merge(test_feat[["row_id"] + target_cols], on="row_id", how="left")

print(f"\nShape: {sub.shape}")
print(f"NaN: {sub.isna().sum().sum()}")
for col in target_cols:
    print(f"  {col}: [{sub[col].min():.4f}, {sub[col].max():.4f}]")

sub.to_csv("submission.csv", index=False)
print("\n✅ Saved submission.csv")
print(sub.head(5))
print("\nDone! 🎉")
