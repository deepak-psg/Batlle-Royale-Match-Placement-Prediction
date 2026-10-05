"""
Model Training and Benchmarking Pipeline.
Trains progressively:
1. Dummy Mean Baseline
2. Linear Regression (Ridge)
3. LightGBM Regressor (Primary Model)
4. XGBoost Regressor (Comparison Model)

Evaluates on strictly held-out matches with group-aware validation.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor

from src.models.baselines import DummyMeanPredictor, build_linear_baseline
from src.models.evaluate import evaluate_predictions, print_evaluation_report
from src.features.build_features import get_feature_groups

def get_train_feature_names(exclude_groups: List[str] = None) -> List[str]:
    """Compile numeric feature columns, optionally excluding specified feature groups."""
    groups = get_feature_groups()
    if exclude_groups is None:
        exclude_groups = []
        
    features = []
    for g_name in ['raw_numeric', 'efficiency', 'team_aggregates', 'match_relative']:
        if g_name not in exclude_groups:
            features.extend(groups[g_name])
            
    # Include numeric match context features
    if 'match_context' not in exclude_groups:
        features.extend(['players_in_match', 'groups_in_match', 'group_size'])
        
    # Remove duplicates and leakage features if requested
    if 'leakage_prone' in exclude_groups:
        features = [f for f in features if not any(leak in f for leak in groups['leakage_prone'])]
        
    # Deduplicate while preserving order
    seen = set()
    final_features = []
    for f in features:
        if f not in seen:
            seen.add(f)
            final_features.append(f)
            
    return final_features


def train_and_eval_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    df_val_meta: pd.DataFrame,
    params: Dict[str, Any] = None
) -> Tuple[lgb.LGBMRegressor, Dict[str, Any]]:
    """Train LightGBM with early stopping and evaluate on held-out set."""
    if params is None:
        params = {
            'objective': 'mae',
            'metric': 'mae',
            'n_estimators': 1500,
            'learning_rate': 0.08,
            'num_leaves': 63,
            'max_depth': -1,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        }
        
    feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
    X_train_c = np.ascontiguousarray(X_train, dtype=np.float32)
    y_train_c = np.ascontiguousarray(y_train, dtype=np.float32)
    X_val_c = np.ascontiguousarray(X_val, dtype=np.float32)
    y_val_c = np.ascontiguousarray(y_val, dtype=np.float32)

    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    
    t0 = time.time()
    model.fit(
        X_train_c, y_train_c,
        feature_name=feature_names,
        eval_set=[(X_val_c, y_val_c)],
        callbacks=callbacks
    )
    train_time = time.time() - t0
    
    preds = model.predict(X_val_c)
    eval_results = evaluate_predictions(y_val, preds, df_meta=df_val_meta)
    eval_results['train_time_sec'] = train_time
    eval_results['best_iteration'] = model.best_iteration_
    
    return model, eval_results
