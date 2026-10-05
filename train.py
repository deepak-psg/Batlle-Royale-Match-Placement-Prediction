"""
Model Training Root Runner.
Points to the primary LightGBM training pipeline in scripts/train_lgbm.py.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "train_lgbm.py"
    runpy.run_path(str(target), run_name="__main__")
