"""
Evaluation Metrics and Diagnostic Reporting Module.
Computes MAE and RMSE with detailed breakdowns:
- By match type (squad, duo, solo, custom)
- By match size (player count bins)
- By target placement tier (early exit, mid game, top 10, winners)
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_meta: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Evaluates placement predictions with global and segmented error metrics.
    y_pred is automatically clipped to [0.0, 1.0].
    """
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    
    global_mae = mean_absolute_error(y_true, y_pred_clipped)
    global_rmse = root_mean_squared_error(y_true, y_pred_clipped)
    
    results = {
        "global_mae": float(global_mae),
        "global_rmse": float(global_rmse),
        "by_match_type": {},
        "by_match_size": {},
        "by_target_tier": {}
    }
    
    if df_meta is not None:
        eval_df = df_meta.copy()
        eval_df["y_true"] = y_true
        eval_df["y_pred"] = y_pred_clipped
        eval_df["abs_error"] = np.abs(eval_df["y_true"] - eval_df["y_pred"])
        
        # 1. Breakdown by match type
        if "match_type_group" in eval_df.columns:
            for mtype, group in eval_df.groupby("match_type_group", observed=False):
                results["by_match_type"][str(mtype)] = {
                    "count": int(len(group)),
                    "mae": float(group["abs_error"].mean())
                }
                
        # 2. Breakdown by match size
        if "players_in_match" in eval_df.columns:
            size_bins = pd.cut(
                eval_df["players_in_match"],
                bins=[0, 50, 85, 100],
                labels=["<50 players", "50-85 players", "86-100 players"]
            )
            eval_df["size_tier"] = size_bins
            for stier, group in eval_df.groupby("size_tier", observed=False):
                results["by_match_size"][str(stier)] = {
                    "count": int(len(group)),
                    "mae": float(group["abs_error"].mean())
                }
                
        # 3. Breakdown by target placement percentile range
        tier_bins = pd.cut(
            eval_df["y_true"],
            bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
            labels=["Bottom 25% (0.00-0.25)", "Mid-Low (0.25-0.50)", "Mid-High (0.50-0.75)", "Top 25% (0.75-1.00)"]
        )
        eval_df["target_tier"] = tier_bins
        for ttier, group in eval_df.groupby("target_tier", observed=False):
            results["by_target_tier"][str(ttier)] = {
                "count": int(len(group)),
                "mae": float(group["abs_error"].mean())
            }
            
    return results


def print_evaluation_report(results: Dict[str, Any], model_name: str = "Model"):
    """Format and print an evaluation summary table."""
    print(f"\n==========================================")
    print(f"  EVALUATION REPORT: {model_name}")
    print(f"==========================================")
    print(f"Global MAE  : {results['global_mae']:.5f}")
    print(f"Global RMSE : {results['global_rmse']:.5f}")
    
    if results.get("by_match_type"):
        print("\n--- Breakdown by Match Type ---")
        for mtype, metrics in results["by_match_type"].items():
            print(f"  {mtype:12s} | Players: {metrics['count']:7,d} | MAE: {metrics['mae']:.5f}")
            
    if results.get("by_match_size"):
        print("\n--- Breakdown by Match Size ---")
        for stier, metrics in results["by_match_size"].items():
            print(f"  {stier:14s} | Players: {metrics['count']:7,d} | MAE: {metrics['mae']:.5f}")
            
    if results.get("by_target_tier"):
        print("\n--- Breakdown by Target Range ---")
        for ttier, metrics in results["by_target_tier"].items():
            print(f"  {ttier:22s} | Players: {metrics['count']:7,d} | MAE: {metrics['mae']:.5f}")
    print(f"==========================================\n")
