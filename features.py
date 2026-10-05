"""
Feature engineering for PUBG Finish Placement Prediction.

Takes raw match/player data and produces the engineered feature set
described in the project blueprint:
  - match-size normalization
  - efficiency ratios
  - group (team) aggregates
  - match-relative percentile ranks
"""

import numpy as np
import pandas as pd


def add_players_joined(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many players actually joined each match."""
    df = df.copy()
    df["playersJoined"] = df.groupby("matchId")["matchId"].transform("count")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that look like cheaters / broken telemetry.
    Rule of thumb used by most PUBG Kaggle solutions:
      - kills > 0 but walkDistance == 0 -> impossible without hacking
      - extremely high kills relative to players in the match
    """
    df = df.copy()
    cheater_mask = (df["kills"] > 0) & (df["walkDistance"] == 0)
    df = df[~cheater_mask]

    # Cap unrealistic kill counts (more kills than players that joined)
    df = df[df["kills"] <= df["playersJoined"]]

    return df


def normalize_by_match_size(df: pd.DataFrame) -> pd.DataFrame:
    """Scale kill/damage stats to a common 100-player match size."""
    df = df.copy()
    scale = 100 / df["playersJoined"].clip(lower=1)
    df["killsNorm"] = df["kills"] * scale
    df["damageDealtNorm"] = df["damageDealt"] * scale
    return df


def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios that capture play-style efficiency, not just raw totals."""
    df = df.copy()
    df["healsAndBoosts"] = df["heals"] + df["boosts"]
    df["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    df["damagePerKill"] = df["damageDealt"] / (df["kills"] + 1)
    df["killsPerWalkDistance"] = df["kills"] / (df["walkDistance"] + 1)
    return df


def add_group_aggregates(df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """
    For each team (groupId), compute mean/max of key stats across teammates.
    Placement is a team-level outcome, so teammate performance matters.
    """
    df = df.copy()
    group_stats = df.groupby(["matchId", "groupId"])[stat_cols].agg(["mean", "max"])
    group_stats.columns = [f"group_{col}_{agg}" for col, agg in group_stats.columns]
    group_stats = group_stats.reset_index()
    df = df.merge(group_stats, on=["matchId", "groupId"], how="left")
    return df


def add_match_relative_ranks(df: pd.DataFrame, rank_cols: list[str]) -> pd.DataFrame:
    """
    Convert each group's aggregated stat into a percentile rank WITHIN its
    match. This is the single most important feature set: placement itself
    is a rank, so rank-based features correlate with it far more strongly
    than raw numbers.
    """
    df = df.copy()
    for col in rank_cols:
        rank_col_name = f"{col}_rankPerc"
        df[rank_col_name] = df.groupby("matchId")[col].rank(pct=True)
    return df


def encode_match_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode matchType (solo/duo/squad and fpp variants)."""
    df = df.copy()
    df = pd.get_dummies(df, columns=["matchType"], prefix="matchType")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline, in order."""
    df = add_players_joined(df)
    df = remove_outliers(df)
    df = normalize_by_match_size(df)
    df = add_efficiency_features(df)

    group_agg_cols = [
        "kills", "damageDealt", "walkDistance", "heals", "boosts", "killPlace"
    ]
    df = add_group_aggregates(df, group_agg_cols)

    rank_cols = [c for c in df.columns if c.startswith("group_")]
    df = add_match_relative_ranks(df, rank_cols)

    df = encode_match_type(df)

    return df


DROP_COLS = ["Id", "groupId", "matchId"]


def get_feature_columns(df: pd.DataFrame, target_col: str = "winPlacePerc") -> list[str]:
    """Everything except identifiers and the target is a model feature."""
    exclude = set(DROP_COLS + [target_col])
    return [c for c in df.columns if c not in exclude]
