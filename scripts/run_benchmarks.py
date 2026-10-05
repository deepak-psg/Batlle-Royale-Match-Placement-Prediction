"""
Comprehensive Benchmarking Script for Progressive Models.
Trains and compares:
1. Dummy Mean Baseline
2. Linear Regression (Ridge)
3. XGBoost Regressor
4. LightGBM Regressor (Primary Model)

Evaluates on strictly held-out matches with group-aware validation.
Saves metrics to reports/benchmark_summary.json and best model to models/lgbm_model.joblib.
"""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.baselines import DummyMeanPredictor, build_linear_baseline
from src.models.evaluate import evaluate_predictions, print_evaluation_report
from src.models.train_models import get_train_feature_names, train_and_eval_lgbm
from xgboost import XGBRegressor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_matches", type=int, default=6000,
                        help="Number of training matches to sample for fast benchmark (0 for full)")
    args = parser.parse_args()
    
    features_parquet = PROJECT_ROOT / "data" / "processed" / "features_full.parquet"
    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("[*] Loading Match Split IDs...")
    train_match_ids = np.load(split_dir / "train_match_ids.npy", allow_pickle=True)
    test_match_ids = np.load(split_dir / "test_match_ids.npy", allow_pickle=True)
    
    if args.sample_matches > 0 and args.sample_matches < len(train_match_ids):
        print(f"[*] Subsampling {args.sample_matches} train matches and {args.sample_matches // 4} test matches for rapid benchmarking...")
        train_match_ids = train_match_ids[:args.sample_matches]
        test_match_ids = test_match_ids[:args.sample_matches // 4]
        
    train_set = set(train_match_ids)
    test_set = set(test_match_ids)
    
    print("[*] Reading features data with Polars...")
    t0 = time.time()
    feature_cols = get_train_feature_names()
    meta_cols = ["matchId", "groupId", "matchType", "match_type_group", "players_in_match", "winPlacePerc"]
    all_needed_cols = list(set(feature_cols + meta_cols))
    
    df_pl = pl.read_parquet(features_parquet, columns=all_needed_cols)
    print(f"[+] Loaded data in {time.time() - t0:.2f}s")
    
    print("[*] Filtering train and held-out test splits...")
    df_train = df_pl.filter(pl.col("matchId").is_in(train_set)).to_pandas()
    df_test = df_pl.filter(pl.col("matchId").is_in(test_set)).to_pandas()
    
    print(f"[+] Train dataset: {len(df_train):,} rows across {df_train['matchId'].nunique():,} matches")
    print(f"[+] Held-out Test: {len(df_test):,} rows across {df_test['matchId'].nunique():,} matches")
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train["winPlacePerc"].to_numpy()
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test["winPlacePerc"].to_numpy()
    
    benchmarks = {}
    
    # -------------------------------------------------------------
    # Model 1: Mean Baseline
    # -------------------------------------------------------------
    print("\n>>> Training Model 1: Dummy Mean Baseline...")
    mean_model = DummyMeanPredictor()
    mean_model.fit(df_train, y_train, match_type_col="match_type_group")
    mean_preds = mean_model.predict(df_test, match_type_col="match_type_group")
    mean_eval = evaluate_predictions(y_test, mean_preds, df_meta=df_test)
    print_evaluation_report(mean_eval, "1. Dummy Mean Baseline")
    benchmarks["Mean_Baseline"] = mean_eval
    
    # -------------------------------------------------------------
    # Model 2: Linear Regression (Ridge)
    # -------------------------------------------------------------
    print("\n>>> Training Model 2: Linear Regression (Ridge Baseline)...")
    linear_pipe = build_linear_baseline()
    t_lin = time.time()
    linear_pipe.fit(X_train, y_train)
    t_lin_total = time.time() - t_lin
    lin_preds = linear_pipe.predict(X_test)
    lin_eval = evaluate_predictions(y_test, lin_preds, df_meta=df_test)
    lin_eval["train_time_sec"] = t_lin_total
    print_evaluation_report(lin_eval, "2. Ridge Linear Regression")
    benchmarks["Ridge_Regression"] = lin_eval
    
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Model 3: LightGBM Regressor (Primary Model)
    # -------------------------------------------------------------
    print("\n>>> Training Model 3: LightGBM Regressor (Primary Model)...")
    lgbm_model, lgbm_eval = train_and_eval_lgbm(
        X_train, y_train,
        X_test, y_test,
        df_val_meta=df_test
    )
    print_evaluation_report(lgbm_eval, "3. LightGBM Regressor (Primary)")
    benchmarks["LightGBM"] = lgbm_eval
    
    # Save best primary model
    best_model_path = models_dir / "lgbm_model.joblib"
    joblib.dump({
        "model": lgbm_model,
        "features": feature_cols,
        "meta": {"trained_rows": len(df_train), "mae": lgbm_eval["global_mae"]}
    }, best_model_path)
    print(f"[+] Saved trained LightGBM model artifact to {best_model_path}")

    # -------------------------------------------------------------
    # Model 4: XGBoost Regressor (Comparison Model)
    # -------------------------------------------------------------
    print("\n>>> Training Model 4: XGBoost Regressor (Comparison Model)...")
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method="hist"
    )
    t_xgb = time.time()
    xgb_model.fit(X_train, y_train)
    t_xgb_total = time.time() - t_xgb
    xgb_preds = xgb_model.predict(X_test)
    xgb_eval = evaluate_predictions(y_test, xgb_preds, df_meta=df_test)
    xgb_eval["train_time_sec"] = t_xgb_total
    print_evaluation_report(xgb_eval, "4. XGBoost Regressor")
    benchmarks["XGBoost"] = xgb_eval
    
    # Save benchmark summary JSON
    summary_path = reports_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"[+] Benchmark summary saved to {summary_path}")
    
    # Summary Comparison Table
    print("\n=======================================================")
    print(f"       FINAL BENCHMARK COMPARISON TABLE")
    print("=======================================================")
    print(f"{'Model':<22} | {'Held-out MAE':<14} | {'Held-out RMSE':<14} | {'Train Time (s)':<14}")
    print("-" * 72)
    for name, res in benchmarks.items():
        t_sec = res.get("train_time_sec", 0.0)
        print(f"{name:<22} | {res['global_mae']:<14.5f} | {res['global_rmse']:<14.5f} | {t_sec:<14.2f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
