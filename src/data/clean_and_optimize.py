"""
Module for cleaning, anomaly removal, and memory optimization of PUBG match data.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric datatypes to minimize memory footprint."""
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
                    
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)
            
        elif col == "matchType":
            df[col] = df[col].astype("category")
            
    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if verbose:
        reduction = 100 * (1 - end_mem / start_mem)
        print(f"[Memory] Reduced memory footprint from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
        
    return df


def remove_anomalies(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove known anomalous and impossible records:
    1. Null target rows (the known 1 row in train_V2)
    2. Players with kills > 0 but total movement distance == 0
    3. Blatant aimbotters: >= 10 kills with 100% headshot accuracy
    4. Impossible longest kill > 1,000m
    5. Absurd weapon acquisitions (> 50)
    """
    initial_len = len(df)
    
    # 1. Drop null target
    df = df.dropna(subset=['winPlacePerc']).copy()
    
    # Calculate total travel distance
    total_distance = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    
    # Filter conditions
    c_zero_dist_kills = (total_distance == 0) & (df['kills'] > 0)
    c_aimbot = (df['kills'] >= 10) & (df['headshotKills'] == df['kills'])
    c_extreme_longest_kill = df['longestKill'] > 1000
    c_extreme_weapons = df['weaponsAcquired'] > 50
    
    mask_to_remove = c_zero_dist_kills | c_aimbot | c_extreme_longest_kill | c_extreme_weapons
    
    removed_count = mask_to_remove.sum()
    df_clean = df[~mask_to_remove].copy()
    
    if verbose:
        print(f"[Cleaning] Removed {removed_count:,} anomalous / cheater rows ({removed_count / initial_len * 100:.3f}%).")
        print(f"[Cleaning] Remaining valid rows: {len(df_clean):,}")
        
    return df_clean


def clean_and_save_data(
    raw_csv_path: str,
    output_parquet_path: str,
    verbose: bool = True
) -> pd.DataFrame:
    """Load raw dataset, clean anomalies, reduce memory, and save to high-speed Parquet."""
    if verbose:
        print(f"[*] Reading raw CSV from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    
    if verbose:
        print("[*] Filtering anomalies and nulls...")
    df_clean = remove_anomalies(df, verbose=verbose)
    
    if verbose:
        print("[*] Downcasting datatypes...")
    df_opt = reduce_mem_usage(df_clean, verbose=verbose)
    
    out_path = Path(output_parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[*] Exporting cleaned dataset to Parquet: {out_path}...")
    df_opt.to_parquet(out_path, index=False, engine='pyarrow')
    
    if verbose:
        file_size_mb = out_path.stat().st_size / (1024 ** 2)
        print(f"[+] Successfully saved Parquet ({file_size_mb:.2f} MB).")
        
    return df_opt
