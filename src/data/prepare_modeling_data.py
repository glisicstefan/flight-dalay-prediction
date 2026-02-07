"""
Prepare modeling data - separate features (X) from target (y).

This script:
1. Loads processed train/val/test datasets
2. Separates features (X) from target (y)
3. Validates data quality using DataValidator
4. Saves X and y as separate CSV files for modeling
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))
from data.validate_data import DataValidator


def load_and_split(filepath: str) -> tuple:
    """
    Load processed data and split into features (X) and target (y).
    
    Args:
        filepath: Path to processed CSV file
        
    Returns:
        X: Features DataFrame (all columns except ARRIVAL_DELAY)
        y: Target Series (ARRIVAL_DELAY)
    """
    data = pd.read_csv(filepath)
    
    X = data.drop('ARRIVAL_DELAY', axis=1)
    y = data['ARRIVAL_DELAY']
    
    return X, y


def print_summary(X: pd.DataFrame, y: pd.Series, name: str):
    """
    Print summary statistics for X and y.
    
    Args:
        X: Features DataFrame
        y: Target Series
        name: Dataset name (e.g., "Train", "Validation")
    """
    print(f"\n{'='*60}")
    print(f"{name} Dataset Summary")
    print(f"{'='*60}")
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Missing values in X: {X.isnull().sum().sum()}")
    print(f"Missing values in y: {y.isnull().sum().sum()}")
    print(f"\nTarget (ARRIVAL_DELAY) statistics:")
    print(y.describe())
    
    non_numeric = X.select_dtypes(exclude=['number'])
    if len(non_numeric) > 0:
        print(f"\n⚠️  Non-numeric columns found: {non_numeric.columns.tolist()}")
    else:
        print(f"\n✅ All features are numeric")


def save_data(X: pd.DataFrame, y: pd.Series, prefix: str, output_dir: str = "../../data/processed"):
    """
    Save X and y as separate CSV files.
    
    Args:
        X: Features DataFrame
        y: Target Series
        prefix: Filename prefix (e.g., "train", "val", "test")
        output_dir: Directory to save files
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X.to_csv(os.path.join(output_dir, f"X_{prefix}.csv"), index=False)
    y.to_csv(os.path.join(output_dir, f"y_{prefix}.csv"), index=False, header=['ARRIVAL_DELAY'])
    
    print(f"✅ Saved {prefix} data:")
    print(f"   X_{prefix}.csv: {X.shape}")
    print(f"   y_{prefix}.csv: {y.shape}")


def main():
    """Main execution function."""
    print("="*60)
    print("PREPARING MODELING DATA")
    print("="*60)
    
    # Load and split data
    print("\nStep 1: Loading and splitting data...")
    X_train, y_train = load_and_split(filepath="../../data/processed/train.csv")
    X_val, y_val = load_and_split(filepath="../../data/processed/val.csv")
    X_test, y_test = load_and_split(filepath="../../data/processed/test.csv")
    
    # Print summaries
    print("\nStep 2: Data summaries...")
    print_summary(X_train, y_train, name="Train")
    print_summary(X_val, y_val, name="Validation")
    print_summary(X_test, y_test, name="Test")
    
    # Validate data quality
    print("\nStep 3: Validating data quality...")
    
    # Fit DataValidator na train data
    validator = DataValidator()
    validator.fit(X_train)
    
    # Validate val and test sets
    val_report = validator.validate(X_val, "Validation")
    test_report = validator.validate(X_test, "Test")
    
    # Print validation reports
    validator.print_report(val_report)
    validator.print_report(test_report)
    
    # Save data
    print("\nStep 4: Saving X and y as separate CSV files...")
    
    # Save train, val, test data
    save_data(X_train, y_train, "train")
    save_data(X_val, y_val, "val")
    save_data(X_test, y_test, "test")
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check data/processed/ for X_*.csv and y_*.csv files")
    print("  2. Ready to start model training!")


if __name__ == "__main__":
    main()