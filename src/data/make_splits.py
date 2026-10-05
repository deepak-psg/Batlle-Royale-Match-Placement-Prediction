"""
Group-Aware Data Splitting Module.
Ensures that data is split strictly by matchId (no match appears in both train and test),
preventing match-level data leakage.
"""

from pathlib import Path
from typing import Tuple, Generator
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def split_matches(
    match_ids: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split unique match IDs into train/validation and held-out test sets.
    """
    unique_matches = np.unique(match_ids)
    rng = np.random.default_rng(seed=random_state)
    shuffled_matches = rng.permutation(unique_matches)
    
    n_test = int(len(shuffled_matches) * test_size)
    test_matches = shuffled_matches[:n_test]
    train_matches = shuffled_matches[n_test:]
    
    return train_matches, test_matches


def get_group_cv_folds(
    df: pd.DataFrame,
    group_col: str = 'matchId',
    n_splits: int = 5,
    random_state: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate GroupKFold cross-validation indices based on matchId.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col]
    X_dummy = np.zeros(len(df))
    
    for train_idx, val_idx in gkf.split(X_dummy, groups=groups):
        yield train_idx, val_idx
