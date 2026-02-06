"""
Data preparation pipeline.

This script orchestrates the complete data preparation workflow:
1. Load cleaned data
2. Create train/validation/test splits
3. Engineer features
4. Save processed datasets and fitted feature engineer

Usage:
    python src/data/prepare_data.py
"""

import pandas as pd
import os
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from features.build_features import AirportFeatureEngineer
from data.make_dataset import create_random_splits
import joblib


def prepare_data(
    input_path="../../data/interim/flights_cleaned.csv",
    output_dir="../../data/processed/",
    models_dir="../../models/"
):
    """
    Complete data preparation pipeline.
    
    Args:
        input_path: Path to cleaned dataset
        output_dir: Directory to save processed data
        models_dir: Directory to save fitted feature engineer
    
    Returns:
        None (saves files to disk)
    """
    
    print("="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    create_random_splits(input_file_path=input_path, output_dir=output_dir)
    print("\n[1/4] Creating train/validation/test splits...")
    
    train = pd.read_csv(os.path.join(output_dir, "train.csv"))
    val = pd.read_csv(os.path.join(output_dir, "val.csv"))
    test = pd.read_csv(os.path.join(output_dir, "test.csv"))
    print("\n[2/4] Loading splits...")
    
    fe = AirportFeatureEngineer()
    fe.fit(train)
    train_transf = fe.transform(train)
    val_transf = fe.transform(val)
    test_transf = fe.transform(test)
    print("\n[3/4] Engineering features...")


    joblib.dump(fe, os.path.join(models_dir, "feature_engineer.pkl"))
    train_transf.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_transf.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_transf.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("\n[4/4] Saving processed data and feature engineer...")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nDataset Shapes:")
    print(f"{'Dataset':<12} {'Rows':>12} {'Columns':>10}")
    print("-" * 37)
    print(f"{'Train':<12} {train_transf.shape[0]:>12,} {train_transf.shape[1]:>10}")
    print(f"{'Validation':<12} {val_transf.shape[0]:>12,} {val_transf.shape[1]:>10}")
    print(f"{'Test':<12} {test_transf.shape[0]:>12,} {test_transf.shape[1]:>10}")

    print("\nFiles Saved:")
    print(f"  - {output_dir}train.csv")
    print(f"  - {output_dir}val.csv")
    print(f"  - {output_dir}test.csv")
    print(f"  - {models_dir}feature_engineer.pkl")


if __name__ == "__main__":
    prepare_data()