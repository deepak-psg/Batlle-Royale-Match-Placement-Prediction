"""
SHAP Interpretability and Feature Importance Analysis.
Calculates global and local feature attributions using TreeExplainer on LightGBM.
Generates attribution summaries and exports plots.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

def compute_shap_explanations(
    model: Any,
    X_sample: pd.DataFrame,
    reports_dir: Path,
    max_display: int = 20
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Computes SHAP values using TreeExplainer on a representative background sample.
    Saves summary plots and feature importance rankings.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Computing Tree SHAP attributions on {len(X_sample)} player records...")
    X_arr64 = np.ascontiguousarray(X_sample.values, dtype=np.float64)
    booster = model.booster_ if hasattr(model, 'booster_') else model
    contribs = booster.predict(X_arr64, pred_contrib=True)
    shap_values = contribs[:, :-1]
    
    # Calculate mean absolute SHAP importance
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    # Save CSV
    importance_csv = reports_dir / "shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    print(f"[+] Saved SHAP importance table to: {importance_csv}")
    
    # Generate and save beeswarm/summary plot
    import shap
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    summary_plot_path = reports_dir / "shap_summary_plot.png"
    plt.savefig(summary_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved SHAP summary plot to: {summary_plot_path}")
    
    # Generate bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    bar_plot_path = reports_dir / "shap_bar_importance.png"
    plt.savefig(bar_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved SHAP bar plot to: {bar_plot_path}")
    
    return shap_values, importance_df
