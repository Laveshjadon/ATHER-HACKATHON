"""
Applies the trained LightGBM model (model/wbt_lgbm.joblib) across context.csv +
test.csv to produce:
  1. submission.csv           -- Kaggle-format 10-day-ahead WBT forecast per row
  2. app/data/predictions.parquet -- compact file the Streamlit app reads

Run from the `ather hackathon` project root, after model/train_model.py:
    py model/score_test.py
"""
import json
import numpy as np
import pandas as pd
import joblib

from train_model import add_base_features, add_lag_features, FEATURE_COLS, DAY_WEIGHTS


def main():
    model = joblib.load("model/wbt_lgbm.joblib")
    with open("model/feature_cols.json") as f:
        feature_cols = json.load(f)

    print("Loading data/context.csv + data/test.csv ...")
    context = pd.read_csv("data/context.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    context["date"] = pd.to_datetime(context["date"])
    context = context.sort_values(["rel_lat", "rel_lon", "date"]).reset_index(drop=True)
    context["day_index"] = context.groupby(["rel_lat", "rel_lon"]).cumcount() - 30

    test = test.sort_values(["rel_lat", "rel_lon", "day_index"]).reset_index(drop=True)

    keep_test = ["row_id", "rel_lat", "rel_lon", "day_index"] + [
        c for c in test.columns if c not in ["row_id", "rel_lat", "rel_lon", "day_index"]]
    keep_ctx = ["row_id", "rel_lat", "rel_lon", "day_index", "WBT"] + [
        c for c in context.columns
        if c not in ["row_id", "rel_lat", "rel_lon", "day_index", "date", "WBT"]]

    combined = pd.concat([context[keep_ctx], test[keep_test]], ignore_index=True)
    combined = combined.sort_values(["rel_lat", "rel_lon", "day_index"]).reset_index(drop=True)
    print(f"Combined (context+test): {combined.shape}")

    combined = add_base_features(combined)
    combined = add_lag_features(combined, ["rel_lat", "rel_lon"])

    test_feat = combined[combined["day_index"] >= 0].copy().reset_index(drop=True)
    for col in feature_cols:
        if col not in test_feat.columns:
            test_feat[col] = 0.0
        test_feat[col] = test_feat[col].fillna(0)

    print("Predicting WBT for every test-period day ...")
    test_feat["WBT_pred"] = model.predict(test_feat[feature_cols].values)

    target_cols = [f"target_day_{d}" for d in range(1, 11)]
    for d in range(1, 11):
        test_feat[f"target_day_{d}"] = test_feat.groupby(
            ["rel_lat", "rel_lon"])["WBT_pred"].shift(-d)
    for col in target_cols:
        test_feat[col] = test_feat.groupby(["rel_lat", "rel_lon"])[col].ffill()
        test_feat[col] = test_feat.groupby(["rel_lat", "rel_lon"])[col].bfill()

    sub = sample_sub[["row_id"]].merge(
        test_feat[["row_id"] + target_cols], on="row_id", how="left")
    import os
    os.makedirs("results", exist_ok=True)
    sub.to_csv("results/submission.csv", index=False)
    print(f"Saved results/submission.csv  shape={sub.shape}  NaN={sub.isna().sum().sum()}")

    # ---- compact artifact for the Streamlit app ----
    test_feat["loc"] = test_feat["row_id"].str.split("_").str[0]
    app_cols = ["row_id", "loc", "rel_lat", "rel_lon", "day_index",
                "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WBT_pred"] + target_cols
    app_df = test_feat[app_cols].copy()
    app_df["max_forecast_wbt"] = app_df[target_cols].max(axis=1)

    def risk_level(w):
        if w >= 35:
            return "Fatal"
        if w >= 32:
            return "Danger"
        if w >= 28:
            return "Caution"
        return "Safe"

    app_df["risk_level"] = app_df["max_forecast_wbt"].apply(risk_level)

    import os
    os.makedirs("app/data", exist_ok=True)
    app_df.to_parquet("app/data/predictions.parquet", index=False)
    print(f"Saved app/data/predictions.parquet  shape={app_df.shape}")

    # small per-district lookup for the sidebar selector
    districts = app_df[["loc", "rel_lat", "rel_lon"]].drop_duplicates().sort_values("loc")
    districts.to_csv("app/data/districts.csv", index=False)
    print(f"Saved app/data/districts.csv  ({len(districts)} districts)")


if __name__ == "__main__":
    main()
