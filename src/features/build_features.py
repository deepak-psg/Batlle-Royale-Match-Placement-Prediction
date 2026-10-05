"""
Feature Engineering Pipeline for PUBG Placement Prediction.
Engineers 4 distinct feature families using high-performance Polars:
1. Efficiency ratios (ratios of actions to distance/kills)
2. Match context (match type clustering, player count, group count)
3. Team aggregates (mean, max, min, sum across teammates)
4. Match-relative features (scaling by match mean, max, and rank percentile)
"""

from typing import List, Dict, Tuple
import polars as pl
import numpy as np
import pandas as pd

# Core raw numeric features
RAW_NUMERIC_FEATURES = [
    'assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills',
    'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks',
    'longestKill', 'matchDuration', 'maxPlace', 'numGroups',
    'rankPoints', 'revives', 'rideDistance', 'roadKills',
    'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
    'weaponsAcquired', 'winPoints'
]

# High-leakage / survival-consequence features identified for ablation analysis
LEAKAGE_PRONE_FEATURES = [
    'walkDistance', 'boosts', 'heals', 'weaponsAcquired'
]

def engineer_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Engineers all 4 feature groups using vectorized Polars operations.
    Fast execution on 4.4M+ rows without memory exhaustion.
    """
    print("[*] Engineering Group 1: Efficiency Ratios...")
    total_distance_expr = pl.col('walkDistance') + pl.col('rideDistance') + pl.col('swimDistance')
    
    df = df.with_columns([
        total_distance_expr.alias('totalDistance'),
        (pl.col('damageDealt') / (pl.col('kills') + 1.0)).alias('damage_per_kill'),
        (pl.col('headshotKills') / (pl.col('kills') + 1.0)).alias('headshot_rate'),
        (pl.col('kills') / (pl.col('walkDistance') + 1.0)).alias('kills_per_walkDistance'),
        (pl.col('heals') / (pl.col('walkDistance') + 1.0)).alias('heals_per_walkDistance'),
        (pl.col('boosts') / (pl.col('walkDistance') + 1.0)).alias('boosts_per_walkDistance'),
        (pl.col('heals') + pl.col('boosts')).alias('heals_and_boosts'),
        ((pl.col('heals') + pl.col('boosts') + pl.col('weaponsAcquired')) / (pl.col('walkDistance') + 1.0)).alias('items_per_walkDistance'),
        (pl.col('kills') / (total_distance_expr + 1.0)).alias('kills_per_totalDistance')
    ])
    
    print("[*] Engineering Group 2: Match Context...")
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
    
    print("[*] Engineering Group 3: Team Aggregates...")
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
    
    print("[*] Engineering Group 4: Match-Relative Scaling & Percentiles...")
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
        # Ratio to match max
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
    
    return df


def get_feature_groups() -> Dict[str, List[str]]:
    """Returns mapping of feature groups to column names for ablation experiments."""
    efficiency_features = [
        'totalDistance', 'damage_per_kill', 'headshot_rate',
        'kills_per_walkDistance', 'heals_per_walkDistance',
        'boosts_per_walkDistance', 'heals_and_boosts',
        'items_per_walkDistance', 'kills_per_totalDistance'
    ]
    
    match_context_features = [
        'players_in_match', 'groups_in_match', 'group_size', 'match_type_group'
    ]
    
    team_stat_cols = [
        'walkDistance', 'totalDistance', 'damageDealt', 'kills',
        'boosts', 'heals', 'weaponsAcquired', 'DBNOs', 'killPlace'
    ]
    team_aggregates = []
    for col in team_stat_cols:
        for stat in ['mean', 'max', 'min', 'sum']:
            team_aggregates.append(f'{col}_team_{stat}')
            
    match_relative_cols = [
        'walkDistance', 'totalDistance', 'damageDealt', 'kills',
        'boosts', 'heals', 'weaponsAcquired', 'killPlace'
    ]
    match_relative = []
    for col in match_relative_cols:
        for rel in ['mean_ratio', 'max_ratio', 'rank_perc']:
            match_relative.append(f'{col}_match_{rel}')
            
    return {
        'raw_numeric': RAW_NUMERIC_FEATURES,
        'efficiency': efficiency_features,
        'match_context': match_context_features,
        'team_aggregates': team_aggregates,
        'match_relative': match_relative,
        'leakage_prone': LEAKAGE_PRONE_FEATURES
    }
