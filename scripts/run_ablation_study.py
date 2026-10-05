"""
Ablation Study and Leakage Analysis Pipeline.
Systematically evaluates model performance when removing:
1. Team Aggregates
2. Match-Relative Features
3. Efficiency Ratios
4. All Engineered Features (Raw Only)
5. Leakage-Prone Features (walkDistance, boosts, heals, weaponsAcquired)

Quantifies the exact contribution of each feature group to validate design choices.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features.build_features import get_feature_groups
from src.models.train_models import get_train_feature_names, train_and_eval_lgbm
from src.models.evaluate import evaluate_predictions

def run_ablation_experiments(sample_matches: int = 3000):
    features_parquet = PROJECT_ROOT / "data" / "processed" / "features_full.parquet"
    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("[*] Loading Match Split IDs for Ablation...")
    train_match_ids = np.load(split_dir / "train_match_ids.npy", allow_pickle=True)[:sample_matches]
    test_match_ids = np.load(split_dir / "test_match_ids.npy", allow_pickle=True)[:sample_matches // 4]
    
    train_set = set(train_match_ids)
    test_set = set(test_match_ids)
    
    groups = get_feature_groups()
    all_feature_cols = get_train_feature_names()
    meta_cols = ["matchId", "groupId", "matchType", "match_type_group", "players_in_match", "winPlacePerc"]
    
    print("[*] Reading Parquet dataset for Ablation...")
    df_pl = pl.read_parquet(features_parquet, columns=list(set(all_feature_cols + meta_cols)))
    df_train = df_pl.filter(pl.col("matchId").is_in(train_set)).to_pandas()
    df_test = df_pl.filter(pl.col("matchId").is_in(test_set)).to_pandas()
    
    y_train = df_train["winPlacePerc"].to_numpy()
    y_test = df_test["winPlacePerc"].to_numpy()
    
    # Define experiment configurations
    configs = {
        "1. Full Feature Set": all_feature_cols,
        "2. Ablation: No Team Aggregates": [f for f in all_feature_cols if not any(f.endswith(f"_team_{s}") for s in ['mean', 'max', 'min', 'sum'])],
        "3. Ablation: No Match-Relative": [f for f in all_feature_cols if not any(f.endswith(f"_match_{s}") for s in ['mean_ratio', 'max_ratio', 'rank_perc'])],
        "4. Ablation: No Efficiency Ratios": [f for f in all_feature_cols if f not in groups['efficiency']],
        "5. Baseline: Raw Features Only": groups['raw_numeric'],
        "6. Leakage Study: No Survival Proxies": [f for f in all_feature_cols if not any(leak in f for leak in groups['leakage_prone'])]
    }
    
    ablation_results = {}
    print("\n=======================================================")
    print("           STARTING ABLATION MATRIX RUNS               ")
    print("=======================================================")
    
    for exp_name, feat_cols in configs.items():
        print(f"\n[*] Running: {exp_name} ({len(feat_cols)} features)...")
        X_tr_c = np.ascontiguousarray(df_train[feat_cols].fillna(0), dtype=np.float32)
        y_tr_c = np.ascontiguousarray(y_train, dtype=np.float32)
        X_te_c = np.ascontiguousarray(df_test[feat_cols].fillna(0), dtype=np.float32)
        y_te_c = np.ascontiguousarray(y_test, dtype=np.float32)
        
        t0 = time.time()
        model = lgb.LGBMRegressor(
            objective='mae',
            metric='mae',
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=45,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        model.fit(
            X_tr_c, y_tr_c,
            eval_set=[(X_te_c, y_te_c)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        elapsed = time.time() - t0
        
        preds = model.predict(X_te_c)
        eval_metrics = evaluate_predictions(y_test, preds, df_meta=df_test)
        
        ablation_results[exp_name] = {
            "num_features": len(feat_cols),
            "mae": eval_metrics["global_mae"],
            "rmse": eval_metrics["global_rmse"],
            "train_time_sec": elapsed
        }
        print(f"    -> Held-out MAE: {eval_metrics['global_mae']:.5f} | Time: {elapsed:.2f}s")
        
    # Save results
    out_file = reports_dir / "ablation_study_results.json"
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
        
    print("\n=========================================================================")
    print(f"                       ABLATION STUDY SUMMARY TABLE                      ")
    print("=========================================================================")
    print(f"{'Experiment':<40} | {'Features':<9} | {'Held-out MAE':<13} | {'MAE Delta vs Full':<17}")
    print("-" * 88)
    base_mae = ablation_results["1. Full Feature Set"]["mae"]
    for exp_name, res in ablation_results.items():
        delta = res["mae"] - base_mae
        delta_str = f"+{delta:.5f}" if delta > 0 else f"{delta:.5f}"
        print(f"{exp_name:<40} | {res['num_features']:<9} | {res['mae']:<13.5f} | {delta_str:<17}")
    print("=========================================================================\n")

if __name__ == "__main__":
    run_ablation_experiments(sample_matches=3000)
