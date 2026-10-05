"""
End-to-end Feature Engineering and Group Split Pipeline.
Executes Polars feature transformation on cleaned data and generates
reproducible group-aware train/test splits.
"""

import sys
import time
from pathlib import Path
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features.build_features import engineer_features_polars, get_feature_groups
from src.data.make_splits import split_matches

def main():
    cleaned_parquet = PROJECT_ROOT / "data" / "processed" / "train_cleaned.parquet"
    features_parquet = PROJECT_ROOT / "data" / "processed" / "features_full.parquet"
    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    if not cleaned_parquet.exists():
        print(f"[Error] Cleaned dataset not found at {cleaned_parquet}")
        sys.exit(1)
        
    print(f"[*] Loading cleaned dataset from {cleaned_parquet} with Polars...")
    t0 = time.time()
    df = pl.read_parquet(cleaned_parquet)
    print(f"[+] Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    
    print("[*] Engineering features across all 4 groups...")
    t1 = time.time()
    df_featured = engineer_features_polars(df)
    feat_time = time.time() - t1
    print(f"[+] Feature engineering completed in {feat_time:.2f}s. New total columns: {len(df_featured.columns)}")
    
    print(f"[*] Saving full featured dataset to {features_parquet}...")
    t2 = time.time()
    df_featured.write_parquet(features_parquet, compression="zstd")
    print(f"[+] Saved {features_parquet.stat().st_size / (1024**2):.2f} MB in {time.time() - t2:.2f}s")
    
    # Generate reproducible group-aware match split (80% train, 20% held-out test)
    print("[*] Generating Group-Aware matchId splits (80/20)...")
    unique_matches = df_featured["matchId"].unique().to_numpy()
    train_matches, test_matches = split_matches(unique_matches, test_size=0.20, random_state=42)
    
    np.save(split_dir / "train_match_ids.npy", train_matches)
    np.save(split_dir / "test_match_ids.npy", test_matches)
    print(f"[+] Group Split Summary:")
    print(f"    Total Matches: {len(unique_matches):,}")
    print(f"    Train Matches: {len(train_matches):,} (80%)")
    print(f"    Held-Out Test Matches: {len(test_matches):,} (20%)")
    
    # Feature group summary
    groups = get_feature_groups()
    print("\n--- [Feature Families Summary] ---")
    for group_name, feats in groups.items():
        print(f"  - {group_name:18s}: {len(feats)} features")
        
    print(f"\n[DONE] Pipeline completed in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
