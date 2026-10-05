"""
Optuna Hyperparameter Tuning Module with Group-Aware Cross-Validation.
Optimizes LightGBM hyperparameters on matchId GroupKFold to minimize MAE.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list,
    n_splits: int = 3
) -> float:
    """Optuna objective function calculating mean MAE across group-aware folds."""
    params = {
        'objective': 'mae',
        'metric': 'mae',
        'n_estimators': 300,
        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.12, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 95),
        'max_depth': trial.suggest_int('max_depth', 6, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
        'subsample': trial.suggest_float('subsample', 0.65, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 5.0, log=True),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_maes = []
    
    X_arr = np.ascontiguousarray(X[feature_names].values, dtype=np.float32)
    y_arr = np.ascontiguousarray(y, dtype=np.float32)
    
    for train_idx, val_idx in gkf.split(X_arr, y_arr, groups=groups):
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_va, y_va = X_arr[val_idx], y_arr[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        preds = np.clip(model.predict(X_va), 0.0, 1.0)
        fold_maes.append(mean_absolute_error(y_va, preds))
        
    return float(np.mean(fold_maes))


def run_optuna_tuning(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list,
    n_trials: int = 15
) -> Dict[str, Any]:
    """Runs Optuna hyperparameter study with group cross-validation."""
    print(f"[*] Starting Optuna Study ({n_trials} trials, GroupKFold on matchId)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X, y, groups, feature_names),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n[+] Optuna Optimization Complete!")
    print(f"    Best Group CV MAE: {study.best_value:.5f}")
    print("    Best Parameters:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")
        
    return study.best_params
