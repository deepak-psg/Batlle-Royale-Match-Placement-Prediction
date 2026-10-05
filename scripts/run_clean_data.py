"""
Script to execute data cleaning, anomaly filtering, memory downcasting,
and convert train_V2.csv to high-speed train_cleaned.parquet.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.clean_and_optimize import clean_and_save_data

def main():
    raw_path = PROJECT_ROOT / "data" / "raw" / "train_V2.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "train_cleaned.parquet"
    
    if not raw_path.exists():
        print(f"[Error] Raw data not found at: {raw_path}")
        sys.exit(1)
        
    start_time = time.time()
    print("[*] Starting Data Cleaning and Parquet Export Pipeline...")
    clean_and_save_data(str(raw_path), str(output_path), verbose=True)
    total_time = time.time() - start_time
    print(f"[DONE] Completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
