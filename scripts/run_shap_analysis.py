"""
SHAP Interpretability Runner Script.
Loads trained LightGBM model, extracts background sample of match records,
and generates global and local SHAP feature attribution visualizations.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.explainability.shap_analysis import compute_shap_explanations

def main():
    model_path = PROJECT_ROOT / "models" / "lgbm_model.joblib"
    features_parquet = PROJECT_ROOT / "data" / "processed" / "features_full.parquet"
    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"[Error] Model not found at {model_path}. Train model first.")
        sys.exit(1)
        
    print("[*] Loading LightGBM Model Artifact...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    print("[*] Loading Held-out Test Sample for SHAP Explanations...")
    test_match_ids = np.load(split_dir / "test_match_ids.npy", allow_pickle=True)[:500]
    test_set = set(test_match_ids)
    
    df_pl = pl.read_parquet(features_parquet, columns=feature_cols + ["matchId"])
    df_sample = df_pl.filter(pl.col("matchId").is_in(test_set)).head(2000).to_pandas()
    X_sample = df_sample[feature_cols].fillna(0)
    
    print(f"[+] Sample shape for SHAP analysis: {X_sample.shape}")
    t0 = time.time()
    shap_vals, imp_df = compute_shap_explanations(model, X_sample, reports_dir, max_display=15)
    print(f"[+] SHAP analysis finished in {time.time() - t0:.2f}s")
    
    print("\n--- [Top 15 Most Influential Features (by Mean |SHAP|)] ---")
    for i, r in imp_df.head(15).iterrows():
        print(f"  {i+1:2d}. {r['feature']:<32s}: {r['mean_abs_shap']:.5f}")

if __name__ == "__main__":
    main()
