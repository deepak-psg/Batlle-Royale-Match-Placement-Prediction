"""
Dataset Inspection and Profiling Script for PUBG Placement Prediction.
Performs memory profiling, missing value verification, match-type distribution,
and anomaly/cheater heuristic detection.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

def run_inspection(csv_path: str):
    print(f"[*] Starting dataset inspection on: {csv_path}")
    start_time = time.time()
    
    # 1. Read dataset
    print("[*] Loading CSV into Pandas...")
    df = pd.read_csv(csv_path)
    load_time = time.time() - start_time
    print(f"[+] Loaded {len(df):,} rows and {len(df.columns)} columns in {load_time:.2f}s")
    
    # 2. Memory Usage
    initial_mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[+] Initial RAM Usage: {initial_mem_mb:.2f} MB")
    
    # 3. Missing Values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    print("\n--- [Null / Missing Values] ---")
    if len(null_cols) == 0:
        print("[+] No null values found across all columns.")
    else:
        for col, cnt in null_cols.items():
            print(f"[!] Warning: Column '{col}' has {cnt} null values ({cnt/len(df)*100:.5f}%)")
            
    # 4. Matches and Groups Overview
    num_matches = df['matchId'].nunique()
    num_groups = df['groupId'].nunique()
    print("\n--- [Entity Cardinality] ---")
    print(f"[+] Total Unique Matches (matchId): {num_matches:,}")
    print(f"[+] Total Unique Groups/Teams (groupId): {num_groups:,}")
    print(f"[+] Average Players Per Match: {len(df) / num_matches:.1f}")
    
    # 5. Match Types
    print("\n--- [Match Types Breakdown] ---")
    match_type_counts = df['matchType'].value_counts()
    for mtype, count in match_type_counts.items():
        print(f"  - {mtype:18s}: {count:8,d} ({count/len(df)*100:5.2f}%)")
        
    # 6. Target Distribution (winPlacePerc)
    print("\n--- [Target Summary: winPlacePerc] ---")
    target = df['winPlacePerc'].dropna()
    print(f"  Min: {target.min():.4f} | Max: {target.max():.4f} | Mean: {target.mean():.4f} | Std: {target.std():.4f}")
    print(f"  Zero placements: {(target == 0).sum():,} ({(target == 0).mean()*100:.2f}%)")
    print(f"  Win placements (1.0): {(target == 1).sum():,} ({(target == 1).mean()*100:.2f}%)")
    
    # 7. Anomaly & Cheater Heuristic Scan
    print("\n--- [Anomaly & Outlier Heuristics] ---")
    total_distance = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    
    # Kills without movement
    zero_dist_kills = df[(total_distance == 0) & (df['kills'] > 0)]
    print(f"[!] Kills with ZERO total movement distance: {len(zero_dist_kills):,} players")
    if len(zero_dist_kills) > 0:
        print(f"    Max kills with zero movement: {zero_dist_kills['kills'].max()}")
        
    # High kills with 100% headshot rate
    aimbot_suspects = df[(df['kills'] >= 10) & (df['headshotKills'] == df['kills'])]
    print(f"[!] 100% Headshot rate with >= 10 kills: {len(aimbot_suspects):,} players")
    
    # Impossible longest kill (e.g. > 1000m)
    extreme_longest_kill = df[df['longestKill'] > 1000]
    print(f"[!] Longest kill > 1,000 meters: {len(extreme_longest_kill):,} players (Max: {df['longestKill'].max():.1f}m)")
    
    # Extreme weapons acquired (e.g. > 50)
    extreme_weapons = df[df['weaponsAcquired'] > 50]
    print(f"[!] Extreme weapons acquired (> 50): {len(extreme_weapons):,} players (Max: {df['weaponsAcquired'].max()})")
    
    # Extreme heals/boosts
    extreme_heals = df[df['heals'] > 40]
    print(f"[!] Extreme heals (> 40): {len(extreme_heals):,} players (Max: {df['heals'].max()})")
    
    # 8. Downcasting & Memory Optimization Simulation
    print("\n--- [Memory Optimization Simulation] ---")
    df_opt = df.copy()
    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        if col_type == 'int64':
            c_min = df_opt[col].min()
            c_max = df_opt[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df_opt[col] = df_opt[col].astype(np.uint8)
                elif c_max < 65535:
                    df_opt[col] = df_opt[col].astype(np.uint16)
                else:
                    df_opt[col] = df_opt[col].astype(np.uint32)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_opt[col] = df_opt[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_opt[col] = df_opt[col].astype(np.int16)
                else:
                    df_opt[col] = df_opt[col].astype(np.int32)
        elif col_type == 'float64':
            df_opt[col] = df_opt[col].astype(np.float32)
        elif col == 'matchType':
            df_opt[col] = df_opt[col].astype('category')
            
    optimized_mem_mb = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
    reduction = (1 - (optimized_mem_mb / initial_mem_mb)) * 100
    print(f"[+] Optimized Memory: {optimized_mem_mb:.2f} MB")
    print(f"[+] Memory Reduction: {reduction:.1f}% savings!")
    print("\n[+] Inspection complete.")

if __name__ == "__main__":
    data_path = Path("C:/Users/ShriRaam S/OneDrive/Desktop/battle royale placement prediction/data/raw/train_V2.csv")
    if not data_path.exists():
        print(f"[!] Error: File not found at {data_path}")
        sys.exit(1)
    run_inspection(str(data_path))
