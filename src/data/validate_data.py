"""
Data validation module.

Validates data quality and checks for potential issues before model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataValidator:
    """
    Validates data quality against expectations learned from training data.
    
    Usage:
        # Fit on training data
        validator = DataValidator()
        validator.fit(train_data)
        
        # Validate other datasets
        val_report = validator.validate(val_data, "Validation")
        validator.print_report(val_report)
    """
    
    def __init__(self):
        """Initialize validator with empty expectations."""
        self.expected_columns_ = None
        self.expected_dtypes_ = None
        self.numeric_ranges_ = None  # min/max za svaku numeric kolonu
        self.categorical_values_ = None  # dozvoljena values za kategorije
        self.numeric_stats_ = None  # mean/std za numeric kolone
        
    def fit(self, train_data: pd.DataFrame):
        """
        Learn expectations from training data.
        
        Args:
            train_data: Training dataset to learn expectations from
            
        Returns:
            self (for method chaining)
        """
        print("Learning data expectations from training set...")
        
        self.expected_columns_ = train_data.columns.tolist()
        
        self.expected_dtypes_ = train_data.dtypes.to_dict()
        
        # min/max for numerical columns
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_ranges_ = {}
        for col in numeric_cols:
            self.numeric_ranges_[col] = {
                'min': train_data[col].min(),
                'max': train_data[col].max()
            }
        
        # Unique values for categorical columns
        categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
        self.categorical_values_ = {}
        for col in categorical_cols:
            self.categorical_values_[col] = set(train_data[col].unique())
        
        # Savin mean/std for numerical values
        self.numeric_stats_= {
            'mean': train_data[numeric_cols].mean().to_dict(),
            'std': train_data[numeric_cols].mean().to_dict()
        }
        
        print("✓ Expectations learned successfully!")
        return self
    
    def _dtypes_compatible(self, expected, actual):
        """Check if two dtypes are compatible (e.g., int64 vs int32)."""
        expected_str = str(expected)
        actual_str = str(actual)
        
        # If both are numeric (int or float)
        if ('int' in expected_str and 'int' in actual_str):
            return True
        if ('float' in expected_str and 'float' in actual_str):
            return True
        
        # Otherwise, must match exactly
        return expected_str == actual_str

    def validate(self, data: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Validate dataset against learned expectations.
        
        Args:
            data: Dataset to validate
            dataset_name: Name for reporting (e.g., "Validation", "Test")
            
        Returns:
            Dictionary with validation results and issues found
        """
        if self.expected_columns_ is None:
            raise ValueError("Validator must be fitted before validation! Call .fit() first.")
        
        print(f"\nValidating {dataset_name} set...")
        
        issues = []
        warnings = []
        
        # Check 1 - Schema validation
        # - Missing columns
        missing_cols = set(self.expected_columns_) - set(data.columns)
        if missing_cols:
            issues.append(f"Missing columns {missing_cols}")
        # - Extra columns
        extra_cols = set(data.columns) - set(self.expected_columns_)
        if extra_cols:
            warnings.append(f"Extra columns (not in training): {extra_cols}")
        # - Wrong dtypes
        for col in data.columns:
            if col in self.expected_dtypes_:
                expected_dype = self.expected_dtypes_[col]
                actual_dype = data[col].dtype
                if not self._dtypes_compatible(expected_dype, actual_dype):
                    issues.append(f"Columns {col}: expected {expected_dype}, got {actual_dype}")
        print(f"   ✓ Schema check: {len(missing_cols)} missing cols, {len(extra_cols)} extra cols")

        # Check 2 - Missing values
        # - Count missing per column
        missing_counts = data.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            for col, count in cols_with_missing.items():
                pct = (count / len(data)) * 100
                if pct > 5:  # Critical: >5% missing
                    issues.append(f"Column '{col}': {count} missing values ({pct:.1f}%)")
                else:  # Warning: <5% missing
                    warnings.append(f"Column '{col}': {count} missing values ({pct:.1f}%)")
        print(f"   ✓ Missing values check: {len(cols_with_missing)} columns with missing data")

        # Check 3 - Value ranges (numeric)
        # - Values outside train min/max
        for col, ranges in self.numeric_ranges_.items():
            if col in data.columns:
                col_min = data[col].min()
                col_max = data[col].max()
                if col_min < ranges['min']:
                    warnings.append(f"Column {col}: min value {col_min:.2f} < training min {ranges['min']:.2f}")
                if col_max > ranges['max']:
                    warnings.append(f"Column {col}: max value {col_max:.2f} > training max {ranges['max']:.2f}")
        print(f"   ✓ Value range check completed")

        # TODO: Check 4 - Unseen categories
        # - Categories not in train set (CRITICAL for AIRLINE, AIRPORT)
        critical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
        for col in critical_cols:
            if col in self.categorical_values_ and col in data.columns:
                train_values = self.categorical_values_[col]
                data_values = set(data[col].unique())

                unseen = data_values - train_values

                if unseen:
                    issues.append(
                        f"Column '{col}': {len(unseen)} unseen categories (not in training): {list(unseen)[:5]}..."
                    )
        print(f"   ✓ Categorical check: {len([i for i in issues if 'unseen' in i])} unseen category issues")

        # TODO: Check 5 - Distribution drift
        # - Mean/std significantly different from train
        for col, stats in self.numeric_stats_.items():
            if col in data.columns:
                data_mean = data[col].mean()
                data_std = data[col].std()
                
                # Check if mean shifted significantly (>20%)
                mean_diff_pct = abs(data_mean - stats['mean']) / stats['mean'] * 100
                
                if mean_diff_pct > 20:
                    warnings.append(
                        f"Column '{col}': mean shifted {mean_diff_pct:.1f}% "
                        f"(train: {stats['mean']:.2f}, current: {data_mean:.2f})"
                    )
        print(f"   ✓ Distribution drift check completed")
        
        return {
            'dataset_name': dataset_name,
            'n_rows': len(data),
            'n_cols': len(data.columns),
            'issues': issues,  # CRITICAL problems
            'warnings': warnings,  # Minor concerns
            'passed': len(issues) == 0
        }
    
    def print_report(self, report: Dict):
        """
        Print formatted validation report.
        
        Args:
            report: Validation report from validate()
        """
        print("\n" + "="*60)
        print(f"VALIDATION REPORT: {report['dataset_name']} Set")
        print("="*60)
        
        # Dataset info
        print("\nDataset Info:")
        print(f"  Rows:      {report['n_rows']:>10,}")
        print(f"  Columns:   {report['n_cols']:>10}")
        
        # Status
        if report['passed']:
            print(f"\nStatus: ✅ PASSED (no critical issues)")
        else:
            print(f"\nStatus: ❌ FAILED ({len(report['issues'])} critical issue(s) found)")
        
        # Issues (critical)
        print(f"\nIssues ({len(report['issues'])}):")
        if len(report['issues']) == 0:
            print("  (none)")
        else:
            for issue in report['issues']:
                print(f"  ❌ {issue}")
        
        # Warnings
        print(f"\nWarnings ({len(report['warnings'])}):")
        if len(report['warnings']) == 0:
            print("  (none)")
        else:
            for warning in report['warnings']:
                print(f"  ⚠ {warning}")
        
        print("\n" + "="*60)


# Test script
if __name__ == "__main__":
    import os
    
    print("="*60)
    print("DATA VALIDATION TEST")
    print("="*60)
    
    # Load processed data
    train = pd.read_csv("../../data/processed/train.csv")
    val = pd.read_csv("../../data/processed/val.csv")
    test = pd.read_csv("../../data/processed/test.csv")
    
    # Fit validator on train
    validator = DataValidator()
    validator.fit(train)
    
    # Validate val and test
    val_report = validator.validate(val, "Validation")
    test_report = validator.validate(test, "Test")
    
    # Print reports
    validator.print_report(val_report)
    validator.print_report(test_report)
    
    # Save validator
    import joblib
    os.makedirs("../../models", exist_ok=True)
    joblib.dump(validator, "../../models/data_validator.pkl")
    print("\n✓ Validator saved to models/data_validator.pkl")