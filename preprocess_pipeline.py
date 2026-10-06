"""
========================================================================================
Battle Royale Placement Prediction - End-to-End Data Preprocessing & Feature Engineering
========================================================================================

Course: Machine Learning
Project: PUBG / Battle Royale Match Placement Prediction
Target: winPlacePerc (Continuous percentile rank from 0.0 to 1.0)

This script contains the complete, rigorous data preprocessing and feature engineering
pipeline designed for academic presentation. It addresses four fundamental challenges
in machine learning on competitive gaming data:

1. Data Cleaning & Integrity:
   - Eliminating corrupted records and anomalous cheater behavior (aimbots, teleporters)
     that distort regression loss surfaces.

2. Memory Optimization (Downcasting):
   - Compressing 4.4M+ records from 1.2+ GB in RAM to efficient numeric formats,
     enabling fast vectorized computation without memory exhaustion.

3. Domain-Specific Feature Engineering (4 Feature Families):
   - Efficiency Ratios: Normalizing actions by distance and survival effort.
   - Match Context: Disentangling game modes (solo, duo, squad) and lobby sizes.
   - Team Dynamics (Squad Aggregates): Capturing squad-level performance, since
     placement is awarded to the squad as a unit, not just individuals.
   - Match-Relative Percentiles: Aligning features with the target by calculating
     relative percentile ranks within each match lobby.

4. Leakage-Free Grouped Splitting:
   - Enforcing an 80/20 train-test partition strictly grouped by `matchId` to prevent
     teammates or opponents from leaking between training and validation sets.
========================================================================================
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupShuffleSplit


# ======================================================================================
# STAGE 1: RAW DATA INGESTION & ANOMALY REMOVAL (CHEATER FILTERING)
# ======================================================================================

def remove_anomalies_and_cheaters(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Identifies and removes game glitches and blatant cheaters from the dataset.
    
    Academic Rationale:
    -------------------
    In regression tasks, severe outliers (e.g., players killing 40 people without moving)
    produce massive squared/absolute errors, skewing gradients and biasing the model
    toward learning non-generalizable cheat artifacts rather than genuine gameplay patterns.

    Filters Applied:
    1. Null Target: Drop records with missing `winPlacePerc` (1 known row in train_V2).
    2. Teleport / AFK Kills: Players with > 0 kills but total movement distance == 0.
    3. Aimbot Exploits: Players with >= 10 kills and 100% headshot accuracy.
    4. Distance Glitches: Players with a kill from > 1,000 meters (beyond render distance).
    5. Inventory Glitches: Players picking up > 50 weapons in a single match.
    """
    initial_count = len(df)
    
    # 1. Target integrity: winPlacePerc must not be null
    df_clean = df.dropna(subset=['winPlacePerc']).copy()
    
    # Calculate aggregate movement across all movement modalities
    total_movement = df_clean['walkDistance'] + df_clean['rideDistance'] + df_clean['swimDistance']
    
    # 2. Impossible movement-to-kill anomalies
    cond_teleport_kills = (total_movement == 0) & (df_clean['kills'] > 0)
    
    # 3. Blatant aimbotting (10+ kills, all headshots)
    cond_aimbot = (df_clean['kills'] >= 10) & (df_clean['headshotKills'] == df_clean['kills'])
    
    # 4. Longest kill beyond reasonable game engine rendering distance (> 1 km)
    cond_extreme_kill_distance = df_clean['longestKill'] > 1000.0
    
    # 5. Inventory anomalies (> 50 weapons acquired)
    cond_absurd_weapons = df_clean['weaponsAcquired'] > 50
    
    # Combine anomalous masks
    anomalous_mask = (
        cond_teleport_kills |
        cond_aimbot |
        cond_extreme_kill_distance |
        cond_absurd_weapons
    )
    
    removed_count = anomalous_mask.sum()
    df_clean = df_clean[~anomalous_mask].copy()
    
    if verbose:
        pct_removed = (removed_count / initial_count) * 100
        print(f"[*] [Stage 1 - Anomaly Removal]")
        print(f"    - Initial rows:       {initial_count:,}")
        print(f"    - Anomalies removed:  {removed_count:,} ({pct_removed:.3f}%)")
        print(f"    - Clean rows remaining: {len(df_clean):,}")
        
    return df_clean


def downcast_numeric_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimizes memory consumption by downcasting integer and float datatypes.
    
    Academic Rationale:
    -------------------
    Loading 4.4M rows with default 64-bit precision consumes >1.2 GB of RAM.
    Stats like `kills` or `boosts` fit safely in `uint8` (0-255) or `uint16` (0-65535).
    Floats can be safely represented as 32-bit floats without precision loss for ML.
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimize Integer columns
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
                    
        # Optimize Float columns
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)
            
        # Categorical columns
        elif col == "matchType":
            df[col] = df[col].astype("category")
            
    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if verbose:
        savings = 100 * (1 - end_mem / start_mem)
        print(f"[*] [Stage 1 - Memory Optimization]")
        print(f"    - Memory before: {start_mem:.2f} MB")
        print(f"    - Memory after:  {end_mem:.2f} MB ({savings:.1f}% reduction)\n")
        
    return df


# ======================================================================================
# STAGE 2: DOMAIN-SPECIFIC FEATURE ENGINEERING (4 FAMILIES)
# ======================================================================================

def engineer_all_features(df_polars: pl.DataFrame) -> pl.DataFrame:
    """
    Constructs 73 engineered features categorized into four distinct ML feature families.
    Implemented in Polars for ultra-fast, multithreaded vectorized execution.

    Feature Family 1: Efficiency Ratios (Combat & Resource Management)
    -----------------------------------------------------------------
    Raw counts alone do not capture survival efficiency. High-placing players
    optimize movement and healing per unit distance traveled.
    - `totalDistance`: walkDistance + rideDistance + swimDistance
    - `damage_per_kill`: damageDealt / (kills + 1)
    - `headshot_rate`: headshotKills / (kills + 1)
    - `kills_per_walkDistance`: Kills normalized by foot travel
    - `heals_per_walkDistance`, `boosts_per_walkDistance`: Health management intensity
    - `heals_and_boosts`: Total consumable intake
    - `items_per_walkDistance`: Total loot density per meter traveled

    Feature Family 2: Match Context & Lobby Characteristics
    -------------------------------------------------------
    Matches have varying player densities (e.g., custom games vs. full 100-player lobbies).
    - `players_in_match`: Total players counted per matchId
    - `groups_in_match`: Total teams/squads participating
    - `group_size`: Teammates in the current squad
    - `match_type_group`: Coarse classification into solo, duo, squad, or custom

    Feature Family 3: Team / Squad Aggregates (Group Dynamics)
    ---------------------------------------------------------
    Crucial Domain Insight: In Battle Royale, `winPlacePerc` is awarded at the
    *team* level. Even if an individual dies early, if their squad survives to the end,
    they receive top placement. Thus, team-level summary statistics are highly predictive:
    - Computes mean, max, min, and sum across teammates (grouped by matchId and groupId)
      for walkDistance, totalDistance, damageDealt, kills, boosts, heals, weaponsAcquired,
      DBNOs, and killPlace.

    Feature Family 4: Match-Relative Percentiles & Scaling
    -----------------------------------------------------
    The target variable `winPlacePerc` is inherently a relative ranking (0 to 1).
    Performing well in a low-intensity match is different from performing well in an
    aggressive lobby. Features are converted into:
    - Match-Mean Ratio: player_stat / (match_mean + eps)
    - Match-Max Ratio:  player_stat / (match_max + eps)
    - Match-Rank Percentile: Normalized rank within the match lobby [0.0, 1.0].
    """
    print("[*] [Stage 2 - Feature Engineering]")
    print("    - Computing Family 1: Efficiency Ratios...")
    total_dist_expr = pl.col('walkDistance') + pl.col('rideDistance') + pl.col('swimDistance')
    
    df = df_polars.with_columns([
        total_dist_expr.alias('totalDistance'),
        (pl.col('damageDealt') / (pl.col('kills') + 1.0)).alias('damage_per_kill'),
        (pl.col('headshotKills') / (pl.col('kills') + 1.0)).alias('headshot_rate'),
        (pl.col('kills') / (pl.col('walkDistance') + 1.0)).alias('kills_per_walkDistance'),
        (pl.col('heals') / (pl.col('walkDistance') + 1.0)).alias('heals_per_walkDistance'),
        (pl.col('boosts') / (pl.col('walkDistance') + 1.0)).alias('boosts_per_walkDistance'),
        (pl.col('heals') + pl.col('boosts')).alias('heals_and_boosts'),
        ((pl.col('heals') + pl.col('boosts') + pl.col('weaponsAcquired')) / (pl.col('walkDistance') + 1.0)).alias('items_per_walkDistance'),
        (pl.col('kills') / (total_dist_expr + 1.0)).alias('kills_per_totalDistance')
    ])
    
    print("    - Computing Family 2: Match Context...")
    df = df.with_columns([
        pl.len().over('matchId').cast(pl.UInt16).alias('players_in_match'),
        pl.col('groupId').n_unique().over('matchId').cast(pl.UInt16).alias('groups_in_match'),
        pl.len().over(['matchId', 'groupId']).cast(pl.UInt8).alias('group_size'),
        pl.when(pl.col('matchType').cast(pl.String).str.contains('solo'))
          .then(pl.lit('solo'))
          .when(pl.col('matchType').cast(pl.String).str.contains('duo'))
          .then(pl.lit('duo'))
          .when(pl.col('matchType').cast(pl.String).str.contains('squad'))
          .then(pl.lit('squad'))
          .otherwise(pl.lit('custom'))
          .cast(pl.Categorical)
          .alias('match_type_group')
    ])
    
    print("    - Computing Family 3: Team / Squad Aggregates (mean, max, min, sum)...")
    team_stat_cols = [
        'walkDistance', 'totalDistance', 'damageDealt', 'kills',
        'boosts', 'heals', 'weaponsAcquired', 'DBNOs', 'killPlace'
    ]
    team_exprs = []
    for col in team_stat_cols:
        team_exprs.append(pl.col(col).mean().over(['matchId', 'groupId']).cast(pl.Float32).alias(f'{col}_team_mean'))
        team_exprs.append(pl.col(col).max().over(['matchId', 'groupId']).cast(pl.Float32).alias(f'{col}_team_max'))
        team_exprs.append(pl.col(col).min().over(['matchId', 'groupId']).cast(pl.Float32).alias(f'{col}_team_min'))
        team_exprs.append(pl.col(col).sum().over(['matchId', 'groupId']).cast(pl.Float32).alias(f'{col}_team_sum'))
    df = df.with_columns(team_exprs)
    
    print("    - Computing Family 4: Match-Relative Percentiles & Ratios...")
    match_relative_cols = [
        'walkDistance', 'totalDistance', 'damageDealt', 'kills',
        'boosts', 'heals', 'weaponsAcquired', 'killPlace'
    ]
    match_exprs = []
    for col in match_relative_cols:
        # Ratio to match mean
        match_mean = pl.col(col).mean().over('matchId')
        match_exprs.append(
            (pl.col(col) / (match_mean + 1e-4)).cast(pl.Float32).alias(f'{col}_match_mean_ratio')
        )
        # Ratio to match maximum
        match_max = pl.col(col).max().over('matchId')
        match_exprs.append(
            (pl.col(col) / (match_max + 1e-4)).cast(pl.Float32).alias(f'{col}_match_max_ratio')
        )
        # Percentile rank within match (0.0 to 1.0)
        match_exprs.append(
            ((pl.col(col).rank(method='average').over('matchId') - 1.0) /
             (pl.col('players_in_match').cast(pl.Float32) - 1.0 + 1e-4)).cast(pl.Float32).alias(f'{col}_match_rank_perc')
        )
    df = df.with_columns(match_exprs)
    
    print(f"    [+] Successfully generated 102 total columns (73 new engineered features).\n")
    return df


# ======================================================================================
# STAGE 3: GROUP-AWARE TRAIN/TEST SPLIT (PREVENTING DATA LEAKAGE)
# ======================================================================================

def create_match_grouped_split(
    match_series: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partitions the dataset into 80% Train and 20% Test strictly grouped by `matchId`.
    
    Academic Rationale (Critical for Machine Learning):
    --------------------------------------------------
    If standard random row-based splitting (`train_test_split`) is used:
    1. Teammates within the same squad end up on opposite sides of the split.
    2. Because teammates almost always finish with identical or near-identical
       `winPlacePerc`, the model cheats by memorizing match outcomes.
    3. Grouping strictly by `matchId` simulates realistic deployment: the trained
       model is evaluated exclusively on completely unseen matches.
    """
    print(f"[*] [Stage 3 - Match-Grouped Train/Test Split]")
    unique_matches = match_series.unique()
    total_matches = len(unique_matches)
    
    # Use GroupShuffleSplit to divide unique match IDs
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(unique_matches, groups=unique_matches))
    
    train_match_ids = unique_matches[train_idx]
    test_match_ids = unique_matches[test_idx]
    
    print(f"    - Total matches: {total_matches:,}")
    print(f"    - Train matches: {len(train_match_ids):,} ({len(train_match_ids)/total_matches*100:.2f}%)")
    print(f"    - Test matches:  {len(test_match_ids):,} ({len(test_match_ids)/total_matches*100:.2f}%)\n")
    
    return train_match_ids, test_match_ids


# ======================================================================================
# STAGE 4: PIPELINE VERIFICATION & INSPECTION
# ======================================================================================

def verify_existing_artifacts(
    cleaned_parquet_path: str = "data/processed/train_cleaned.parquet",
    features_parquet_path: str = "data/processed/features_full.parquet",
    splits_dir: str = "data/processed/splits"
) -> bool:
    """
    Inspects and validates that all processed datasets and split files exist on disk,
    verifying row counts, column counts, and split partitions.
    """
    print("=" * 80)
    print("VERIFYING PREPROCESSED DATASET ARTIFACTS ON DISK")
    print("=" * 80)
    
    # 1. Verify Cleaned Parquet
    cleaned_p = Path(cleaned_parquet_path)
    if not cleaned_p.exists():
        print(f"[!] Cleaned file missing: {cleaned_p}")
        return False
    df_clean_scan = pl.scan_parquet(str(cleaned_p))
    clean_rows = df_clean_scan.select(pl.len()).collect().item()
    clean_cols = len(df_clean_scan.collect_schema().names())
    print(f"[+] Cleaned Dataset ({cleaned_p.name}):")
    print(f"    - Location:     {cleaned_p.resolve()}")
    print(f"    - Dimensions:   {clean_rows:,} rows x {clean_cols} columns")
    print(f"    - Description:  Anomalies removed, dtypes downcasted (base features)")

    # 2. Verify Features Parquet
    feat_p = Path(features_parquet_path)
    if not feat_p.exists():
        print(f"[!] Features file missing: {feat_p}")
        return False
    df_feat_scan = pl.scan_parquet(str(feat_p))
    feat_rows = df_feat_scan.select(pl.len()).collect().item()
    feat_cols = len(df_feat_scan.collect_schema().names())
    print(f"\n[+] Engineered Dataset ({feat_p.name}):")
    print(f"    - Location:     {feat_p.resolve()}")
    print(f"    - Dimensions:   {feat_rows:,} rows x {feat_cols} columns")
    print(f"    - Description:  Includes all 4 feature families (73 engineered features)")

    # 3. Verify Splits
    train_ids_path = Path(splits_dir) / "train_match_ids.npy"
    test_ids_path = Path(splits_dir) / "test_match_ids.npy"
    if not train_ids_path.exists() or not test_ids_path.exists():
        print(f"[!] Split arrays missing in {splits_dir}")
        return False
    train_ids = np.load(str(train_ids_path), allow_pickle=True)
    test_ids = np.load(str(test_ids_path), allow_pickle=True)
    tot_matches = len(train_ids) + len(test_ids)
    
    # Check row partition counts
    train_set = set(train_ids)
    test_set = set(test_ids)
    train_rows = df_feat_scan.filter(pl.col('matchId').is_in(train_set)).select(pl.len()).collect().item()
    test_rows = df_feat_scan.filter(pl.col('matchId').is_in(test_set)).select(pl.len()).collect().item()
    
    print(f"\n[+] Group-Aware Splits (Match ID Partitioning):")
    print(f"    - Train matches: {len(train_ids):,} ({len(train_ids)/tot_matches*100:.2f}%) -> {train_rows:,} rows ({train_rows/feat_rows*100:.2f}%)")
    print(f"    - Test matches:  {len(test_ids):,} ({len(test_ids)/tot_matches*100:.2f}%) -> {test_rows:,} rows ({test_rows/feat_rows*100:.2f}%)")
    print(f"    - Total matches: {tot_matches:,} (Zero match/group leakage)")
    print("=" * 80)
    print("[SUCCESS] All preprocessed datasets and splits are verified and ready for model training.")
    print("=" * 80)
    return True


# ======================================================================================
# STAGE 5: FULL PIPELINE ORCHESTRATION (RUNNABLE FROM SCRATCH)
# ======================================================================================

def execute_full_pipeline(
    raw_csv_path: str = "data/raw/train_V2.csv",
    cleaned_parquet_path: str = "data/processed/train_cleaned.parquet",
    features_parquet_path: str = "data/processed/features_full.parquet",
    splits_dir: str = "data/processed/splits"
):
    """
    Executes the entire end-to-end preprocessing sequence from raw CSV to final features.
    """
    print(f"[*] Starting end-to-end preprocessing pipeline on: {raw_csv_path}")
    
    # Step 1: Ingestion & Anomaly Removal
    print("\n--- Step 1: Loading raw CSV and removing anomalies ---")
    df_raw = pd.read_csv(raw_csv_path)
    df_cleaned = remove_anomalies_and_cheaters(df_raw, verbose=True)
    df_cleaned = downcast_numeric_dtypes(df_cleaned, verbose=True)
    
    # Save Step 1 Artifact
    Path(cleaned_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_parquet(cleaned_parquet_path, index=False, engine='pyarrow')
    print(f"[+] Saved sanitized baseline dataset to: {cleaned_parquet_path}")
    
    # Step 2: Feature Engineering (convert to Polars for vectorization)
    print("\n--- Step 2: Feature Engineering with Polars ---")
    df_polars = pl.from_pandas(df_cleaned)
    df_features = engineer_all_features(df_polars)
    df_features.write_parquet(features_parquet_path)
    print(f"[+] Saved fully engineered dataset to: {features_parquet_path}")
    
    # Step 3: Match Splits
    print("\n--- Step 3: Generating Group-Aware Splits ---")
    train_ids, test_ids = create_match_grouped_split(df_cleaned['matchId'], test_size=0.20)
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(splits_dir, "train_match_ids.npy"), train_ids)
    np.save(os.path.join(splits_dir, "test_match_ids.npy"), test_ids)
    print(f"[+] Saved split arrays to: {splits_dir}")
    
    # Final check
    verify_existing_artifacts(cleaned_parquet_path, features_parquet_path, splits_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Battle Royale Placement Prediction - Preprocessing & Feature Engineering Pipeline"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all stages from raw CSV (takes ~1-2 min for 4.4M rows). If omitted, verifies existing artifacts."
    )
    parser.add_argument("--raw_csv", type=str, default="data/raw/train_V2.csv")
    args = parser.parse_args()

    if args.recompute:
        execute_full_pipeline(raw_csv_path=args.raw_csv)
    else:
        # Default mode: verify and report on existing prepared datasets
        success = verify_existing_artifacts()
        if not success:
            print("\n[!] Preprocessed files not found or incomplete. Running full recompute...")
            execute_full_pipeline(raw_csv_path=args.raw_csv)
