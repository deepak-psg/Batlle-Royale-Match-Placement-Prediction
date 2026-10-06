"""
PUBG Placement Prediction - Model Training Pipeline
Trains Baseline (Linear Regression) & LightGBM with 5-fold GroupKFold on matchId.
"""

import os
import gc
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "8"


def load_train_data(
    features_path: str = "data/processed/features_full.parquet",
    splits_dir: str = "data/processed/splits"
):
    """Load features_full.parquet filtered strictly to train matches."""
    print("[*] Loading train match IDs...")
    train_match_ids = np.load(os.path.join(splits_dir, "train_match_ids.npy"), allow_pickle=True)
    train_match_set = set(train_match_ids)
    print(f"    - Train matches: {len(train_match_ids):,}")

    print("[*] Reading features_full.parquet...")
    t0 = time.time()
    df = pd.read_parquet(features_path)
    df = df[df["matchId"].isin(train_match_set)].reset_index(drop=True)
    print(f"    - Loaded {len(df):,} train rows in {time.time()-t0:.2f}s")

    y = np.ascontiguousarray(df["winPlacePerc"].to_numpy(), dtype=np.float32)
    groups = df["matchId"].to_numpy()

    cat_cols = ["matchType", "match_type_group"]
    num_cols = [c for c in df.columns if c not in cat_cols and c not in ["Id", "groupId", "matchId", "winPlacePerc"]]

    # Assemble C-contiguous numeric feature matrix
    X_num = df[num_cols].to_numpy(dtype=np.float32)
    cat_mat = np.column_stack([df[c].astype("category").cat.codes.to_numpy(dtype=np.float32) for c in cat_cols])
    X_mat = np.ascontiguousarray(np.hstack([X_num, cat_mat]))
    feature_names = num_cols + cat_cols

    del df, X_num, cat_mat
    gc.collect()

    print(f"    - Feature matrix shape: {X_mat.shape} (98 predictors)")
    return X_mat, y, groups, feature_names, cat_cols


def train_linear_regression(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """Train 5-fold GroupKFold Linear Regression baseline."""
    print("\n" + "=" * 60)
    print("TRAINING BASELINE: LINEAR REGRESSION (5-FOLD GROUPKFOLD)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=n_splits)
    maes = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        t0 = time.time()
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        mean = X_tr.mean(axis=0)
        std = X_tr.std(axis=0) + 1e-7
        X_tr_s = (X_tr - mean) / std
        X_va_s = (X_va - mean) / std

        lr = LinearRegression()
        lr.fit(X_tr_s, y_tr)
        preds = np.clip(lr.predict(X_va_s), 0.0, 1.0)
        mae = float(mean_absolute_error(y_va, preds))
        maes.append(mae)

        print(f"  Fold {fold}: MAE = {mae:.5f} ({time.time() - t0:.1f}s)")
        del X_tr, y_tr, X_va, y_va, X_tr_s, X_va_s, lr, preds
        gc.collect()

    mean_mae = float(np.mean(maes))
    print("-" * 60)
    print(f"  Linear Regression Mean 5-Fold MAE: {mean_mae:.5f}")
    print("=" * 60)
    return maes, mean_mae


def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list,
    cat_features: list,
    n_splits: int = 5
):
    """Train 5-fold GroupKFold LightGBM with early stopping."""
    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM REGRESSOR (5-FOLD GROUPKFOLD)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=n_splits)
    maes = []
    best_iters = []
    feature_importances = np.zeros(len(feature_names), dtype=np.float64)

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.10,
        "num_leaves": 31,
        "force_row_wise": True,
        "random_state": 42,
        "n_jobs": 8,
        "verbose": -1,
    }

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        t0 = time.time()
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        dtrain = lgb.Dataset(
            X_tr, label=y_tr,
            feature_name=feature_names,
            categorical_feature=cat_features,
            free_raw_data=True
        )
        dvalid = lgb.Dataset(
            X_va, label=y_va,
            reference=dtrain,
            feature_name=feature_names,
            categorical_feature=cat_features,
            free_raw_data=True
        )

        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=120,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )

        preds = np.clip(bst.predict(X_va), 0.0, 1.0)
        mae = float(mean_absolute_error(y_va, preds))
        maes.append(mae)
        best_iters.append(bst.best_iteration)
        feature_importances += bst.feature_importance(importance_type="gain")

        print(f"  Fold {fold}: MAE = {mae:.5f} (Best Iter: {bst.best_iteration}, {time.time() - t0:.1f}s)")
        del X_tr, y_tr, X_va, y_va, dtrain, dvalid, bst, preds
        gc.collect()

    mean_mae = float(np.mean(maes))
    avg_best_iter = int(np.mean(best_iters))
    feature_importances /= float(n_splits)

    print("-" * 60)
    print(f"  LightGBM Mean 5-Fold MAE: {mean_mae:.5f} (Avg Best Iter: {avg_best_iter})")
    print("=" * 60)

    # Top 10 features by gain importance
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Gain": feature_importances
    }).sort_values("Gain", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("TOP 10 FEATURES BY IMPORTANCE (AVERAGE GAIN)")
    print("=" * 60)
    for i, row in fi_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['Feature']:<35} : {row['Gain']:,.1f}")
    print("=" * 60)

    return maes, mean_mae, avg_best_iter, fi_df


def refit_and_save(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    cat_features: list,
    n_estimators: int,
    model_path: str = "models/lgbm_model.joblib"
):
    """Refit LightGBM on all train matches and save with joblib."""
    print("\n" + "=" * 60)
    print(f"REFITTING LIGHTGBM ON ALL {len(X):,} ROWS ({n_estimators} TREES)")
    print("=" * 60)

    t0 = time.time()
    ds = lgb.Dataset(
        X, label=y,
        feature_name=feature_names,
        categorical_feature=cat_features,
        free_raw_data=True
    )
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.10,
        "num_leaves": 31,
        "force_row_wise": True,
        "random_state": 42,
        "n_jobs": 8,
        "verbose": -1,
    }

    full_bst = lgb.train(params, ds, num_boost_round=n_estimators)
    print(f"[*] Refit completed in {time.time() - t0:.1f}s")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": full_bst,
        "feature_names": feature_names,
        "categorical_features": cat_features,
        "n_estimators": n_estimators
    }, model_path)
    print(f"[+] Model saved to: {model_path}")
    print("=" * 60)


def main():
    # 1. Load data
    X, y, groups, feature_names, cat_cols = load_train_data()

    # 2. Linear Regression Baseline (5-fold GroupKFold)
    lr_maes, lr_mean_mae = train_linear_regression(X, y, groups, n_splits=5)

    # 3. LightGBM (5-fold GroupKFold with early stopping)
    lgb_maes, lgb_mean_mae, avg_best_iter, fi_df = train_lightgbm(
        X, y, groups, feature_names, cat_cols, n_splits=5
    )

    # 4. Final Comparison Table
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON (5-FOLD GROUPKFOLD VALIDATION MAE)")
    print("=" * 60)
    print(f"{'Fold':<10} {'Linear Regression':<22} {'LightGBM':<15}")
    print("-" * 60)
    for i in range(5):
        print(f"Fold {i+1:<5} {lr_maes[i]:<22.5f} {lgb_maes[i]:<15.5f}")
    print("-" * 60)
    print(f"{'Mean MAE':<10} {lr_mean_mae:<22.5f} {lgb_mean_mae:<15.5f}")
    print("=" * 60)
    diff = lr_mean_mae - lgb_mean_mae
    pct = (diff / lr_mean_mae) * 100
    print(f"[*] LightGBM improves upon Linear Regression by {diff:.5f} MAE ({pct:.1f}% error reduction)")
    print("=" * 60)

    # 5. Refit LightGBM on all train matches and save to models/
    refit_and_save(X, y, feature_names, cat_cols, n_estimators=avg_best_iter)


if __name__ == "__main__":
    main()
