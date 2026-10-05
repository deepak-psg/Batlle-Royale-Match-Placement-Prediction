"""
Comprehensive Unit and Integration Test Suite.
Verifies data cleaning, feature engineering, prediction bounds, and FastAPI endpoint.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.clean_and_optimize import remove_anomalies
from src.features.build_features import engineer_features_polars, get_feature_groups
from src.models.evaluate import evaluate_predictions

def test_anomaly_removal():
    """Verify that anomalous and null records are filtered out."""
    raw_sample = pd.DataFrame({
        'winPlacePerc': [0.5, np.nan, 0.8, 0.2],
        'walkDistance': [100.0, 200.0, 0.0, 50.0],
        'rideDistance': [0.0, 0.0, 0.0, 0.0],
        'swimDistance': [0.0, 0.0, 0.0, 0.0],
        'kills': [1, 0, 5, 0],  # row 2 has 5 kills but 0 distance (hacker)
        'headshotKills': [0, 0, 5, 0],
        'longestKill': [20.0, 0.0, 50.0, 10.0],
        'weaponsAcquired': [2, 1, 3, 1]
    })
    
    clean_df = remove_anomalies(raw_sample, verbose=False)
    # Row 1 (nan target) and Row 2 (0 dist with 5 kills) should both be dropped
    assert len(clean_df) == 2
    assert not clean_df['winPlacePerc'].isnull().any()
    assert (clean_df['kills'] == 5).sum() == 0


def test_feature_engineering_polars():
    """Verify that Polars feature engineering produces valid columns without errors."""
    sample_data = {
        'Id': ['p1', 'p2', 'p3', 'p4'],
        'groupId': ['g1', 'g1', 'g2', 'g2'],
        'matchId': ['m1', 'm1', 'm1', 'm1'],
        'assists': [0, 1, 0, 0],
        'boosts': [1, 2, 0, 1],
        'damageDealt': [100.0, 250.0, 50.0, 120.0],
        'DBNOs': [0, 1, 0, 0],
        'headshotKills': [0, 1, 0, 0],
        'heals': [2, 1, 0, 1],
        'killPlace': [40, 20, 60, 35],
        'killPoints': [1000, 1000, 1000, 1000],
        'kills': [1, 2, 0, 1],
        'killStreaks': [1, 1, 0, 1],
        'longestKill': [15.0, 40.0, 0.0, 25.0],
        'matchDuration': [1400, 1400, 1400, 1400],
        'matchType': ['squad-fpp', 'squad-fpp', 'squad-fpp', 'squad-fpp'],
        'maxPlace': [26, 26, 26, 26],
        'numGroups': [25, 25, 25, 25],
        'rankPoints': [1400, 1400, 1400, 1400],
        'revives': [0, 1, 0, 0],
        'rideDistance': [0.0, 100.0, 0.0, 50.0],
        'roadKills': [0, 0, 0, 0],
        'swimDistance': [0.0, 0.0, 0.0, 0.0],
        'teamKills': [0, 0, 0, 0],
        'vehicleDestroys': [0, 0, 0, 0],
        'walkDistance': [500.0, 1200.0, 300.0, 800.0],
        'weaponsAcquired': [3, 5, 2, 4],
        'winPoints': [1000, 1000, 1000, 1000]
    }
    
    pl_df = pl.DataFrame(sample_data)
    feat_df = engineer_features_polars(pl_df)
    
    # Check that new engineered columns exist
    assert 'totalDistance' in feat_df.columns
    assert 'damage_per_kill' in feat_df.columns
    assert 'players_in_match' in feat_df.columns
    assert 'walkDistance_team_mean' in feat_df.columns
    assert 'damageDealt_match_rank_perc' in feat_df.columns
    assert len(feat_df.columns) > 80


def test_evaluate_predictions_bounds():
    """Verify that predictions are clipped to [0.0, 1.0] and MAE is computed."""
    y_true = np.array([0.1, 0.5, 0.9])
    # Unclipped predictions with out-of-bounds values
    y_pred_unclipped = np.array([-0.2, 0.5, 1.4])
    
    results = evaluate_predictions(y_true, y_pred_unclipped)
    assert results['global_mae'] >= 0.0
    # Clipped predictions should be 0.0, 0.5, 1.0 -> absolute errors: 0.1, 0.0, 0.1 -> mean: ~0.0667
    assert np.isclose(results['global_mae'], 0.066666, atol=1e-4)


def test_fastapi_health_endpoint():
    """Test FastAPI /health route."""
    from fastapi.testclient import TestClient
    from app.api import app
    
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
