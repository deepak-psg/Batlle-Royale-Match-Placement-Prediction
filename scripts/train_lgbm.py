"""
Primary LightGBM Model Training and Evaluation Pipeline.
Trains LightGBM regressor on group-split training data and validates
strictly on held-out test matches.
Saves model artifact to models/lgbm_model.joblib and evaluation report.
"""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.evaluate import evaluate_predictions, print_evaluation_report
from src.features.build_features import get_feature_groups
from src.models.train_models import get_train_feature_names

def train_primary_lgbm(sample_matches: int = 4000):
    features_parquet = PROJECT_ROOT / "data" / "processed" / "features_full.parquet"
    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("[*] Loading Match Split IDs...")
    train_match_ids = np.load(split_dir / "train_match_ids.npy", allow_pickle=True)
    test_match_ids = np.load(split_dir / "test_match_ids.npy", allow_pickle=True)
    
    if sample_matches > 0:
        print(f"[*] Subsampling {sample_matches} train matches and {sample_matches // 4} test matches...")
        train_match_ids = train_match_ids[:sample_matches]
        test_match_ids = test_match_ids[:sample_matches // 4]
        
    train_set = set(train_match_ids)
    test_set = set(test_match_ids)
    
    feature_cols = get_train_feature_names()
    meta_cols = ["matchId", "groupId", "matchType", "match_type_group", "players_in_match", "winPlacePerc"]
    
    print(f"[*] Reading feature dataset ({len(feature_cols)} features)...")
    t0 = time.time()
    df_pl = pl.read_parquet(features_parquet, columns=list(set(feature_cols + meta_cols)))
    df_train = df_pl.filter(pl.col("matchId").is_in(train_set)).to_pandas()
    df_test = df_pl.filter(pl.col("matchId").is_in(test_set)).to_pandas()
    print(f"[+] Loaded data in {time.time() - t0:.2f}s")
    print(f"[+] Train: {len(df_train):,} rows | Held-out Test: {len(df_test):,} rows")
    
    X_train = np.ascontiguousarray(df_train[feature_cols].fillna(0).values, dtype=np.float32)
    y_train = np.ascontiguousarray(df_train["winPlacePerc"].values, dtype=np.float32)
    X_test = np.ascontiguousarray(df_test[feature_cols].fillna(0).values, dtype=np.float32)
    y_test = np.ascontiguousarray(df_test["winPlacePerc"].values, dtype=np.float32)
    
    params = {
        'objective': 'mae',
        'metric': 'mae',
        'n_estimators': 1200,
        'learning_rate': 0.08,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': 4,
        'random_state': 42,
        'verbose': -1
    }
    
    print("\n[*] Training Primary LightGBM Regressor with Early Stopping...")
    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    
    t_train = time.time()
    model.fit(
        X_train, y_train,
        feature_name=feature_cols,
        eval_set=[(X_test, y_test)],
        callbacks=callbacks
    )
    train_time = time.time() - t_train
    print(f"[+] Training finished in {train_time:.2f}s (Best Iteration: {model.best_iteration_})")
    
    print("[*] Evaluating on Held-Out Test Set...")
    preds = model.predict(X_test)
    eval_results = evaluate_predictions(y_test, preds, df_meta=df_test)
    eval_results['train_time_sec'] = train_time
    eval_results['best_iteration'] = model.best_iteration_
    print_evaluation_report(eval_results, "Primary LightGBM Regressor")
    
    # Save model artifact
    best_model_path = models_dir / "lgbm_model.joblib"
    joblib.dump({
        "model": model,
        "features": feature_cols,
        "meta": {"trained_rows": len(df_train), "mae": eval_results["global_mae"]}
    }, best_model_path)
    print(f"[+] Model artifact successfully saved to: {best_model_path}")
    
    # Save eval report
    report_file = reports_dir / "lgbm_evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"[+] Evaluation report saved to: {report_file}")
    
    return eval_results

if __name__ == "__main__":
    train_primary_lgbm(sample_matches=4000)
