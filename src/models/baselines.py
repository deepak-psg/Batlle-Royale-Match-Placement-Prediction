"""
Baseline Models for Battle Royale Placement Prediction:
1. DummyMeanPredictor: Predicts empirical mean placement.
2. LinearRegressionBaseline: Scaled Ridge / Linear Regression.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DummyMeanPredictor(BaseEstimator, RegressorMixin):
    """Trivial baseline that predicts global or match-type mean placement."""
    def __init__(self):
        self.mean_val_: float = 0.5
        self.type_means_: Dict[str, float] = {}
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, match_type_col: Optional[str] = "match_type_group"):
        self.mean_val_ = float(np.mean(y))
        if match_type_col and match_type_col in X.columns:
            types = X[match_type_col].astype(str)
            df_temp = pd.DataFrame({"type": types, "target": y})
            self.type_means_ = df_temp.groupby("type", observed=False)["target"].mean().to_dict()
        return self
        
    def predict(self, X: pd.DataFrame, match_type_col: Optional[str] = "match_type_group") -> np.ndarray:
        if self.type_means_ and match_type_col and match_type_col in X.columns:
            types = X[match_type_col].astype(str)
            mapped = types.map(self.type_means_).astype(float).fillna(self.mean_val_)
            return mapped.to_numpy()
        return np.full(len(X), self.mean_val_)


def build_linear_baseline() -> Pipeline:
    """Builds a standardized linear (Ridge) regression baseline pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=10.0, random_state=42))
    ])
