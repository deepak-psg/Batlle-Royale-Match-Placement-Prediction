"""
Train and compare Linear Regression, Random Forest, and LightGBM on the
PUBG placement prediction task.

Usage:
    python src/train.py --data data/train_V2.csv --sample_frac 0.1
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from features import engineer_features, get_feature_columns

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def load_data(path: str, sample_frac: float | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["winPlacePerc"])  # one known bad row in the full dataset
    if sample_frac is not None:
        # Sample by matchId so full matches stay together
        match_ids = df["matchId"].unique()
        n_sample = int(len(match_ids) * sample_frac)
        sampled_matches = np.random.choice(match_ids, n_sample, replace=False)
        df = df[df["matchId"].isin(sampled_matches)]
    return df


def split_by_match(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split train/val by matchId so no group/match leaks across the split."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["matchId"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def clip_predictions(preds: np.ndarray) -> np.ndarray:
    """winPlacePerc is bounded in [0, 1]."""
    return np.clip(preds, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_V2.csv")
    parser.add_argument("--sample_frac", type=float, default=None,
                         help="Fraction of matches to sample, e.g. 0.1 for quick iteration")
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data, args.sample_frac)
    print(f"Loaded {len(df):,} rows across {df['matchId'].nunique():,} matches")

    print("Engineering features ...")
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features")

    train_df, val_df = split_by_match(df)
    X_train, y_train = train_df[feature_cols], train_df["winPlacePerc"]
    X_val, y_val = val_df[feature_cols], val_df["winPlacePerc"]

    results = {}

    # --- Model 1: Linear Regression (needs scaled features) ---
    print("\nTraining Linear Regression ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_val_scaled = scaler.transform(X_val.fillna(0))

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = clip_predictions(lr.predict(X_val_scaled))
    results["Linear Regression"] = mean_absolute_error(y_val, lr_preds)

    # --- Model 2: Random Forest ---
    print("Training Random Forest (this can take a while on the full dataset) ...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
    )
    rf.fit(X_train.fillna(0), y_train)
    rf_preds = clip_predictions(rf.predict(X_val.fillna(0)))
    results["Random Forest"] = mean_absolute_error(y_val, rf_preds)

    # --- Model 3: LightGBM ---
    best_model = rf
    best_name = "Random Forest"
    if HAS_LGBM:
        print("Training LightGBM ...")
        lgbm = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42
        )
        lgbm.fit(X_train, y_train)
        lgbm_preds = clip_predictions(lgbm.predict(X_val))
        results["LightGBM"] = mean_absolute_error(y_val, lgbm_preds)
        if results["LightGBM"] <= min(results.values()):
            best_model, best_name = lgbm, "LightGBM"
    else:
        print("lightgbm not installed — skipping (pip install lightgbm)")

    print("\n=== Validation MAE (lower is better) ===")
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:20s} MAE = {mae:.5f}")

    print(f"\nBest model: {best_name}")
    joblib.dump({"model": best_model, "features": feature_cols}, "best_model.pkl")
    print("Saved best model to best_model.pkl")


if __name__ == "__main__":
    main()
