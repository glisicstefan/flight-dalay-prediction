"""
Heuristic Baseline Models.

Simple rule-based predictions that serve as performance benchmarks
for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class HeuristicBaseline:
    """
    Heuristic baseline models for flight delay prediction.
    
    Strategies:
    1. 'naive_mean': Predict global mean delay for all flights
    2. 'feature_based': Use DESTINATION_AVG_DELAY feature (strongest single feature)
    
    Usage:
        baseline = HeuristicBaseline(strategy='naive_mean')
        baseline.fit(X_train, y_train)
        predictions = baseline.predict(X_val)
        
        metrics = baseline.evaluate(X_val, y_val)
        baseline.print_metrics(metrics)
    """
    
    def __init__(self, strategy='naive_mean'):
        """
        Initialize baseline model.
        
        Args:
            strategy: One of 'naive_mean', 'feature_based'
        """
        self.strategy = strategy
        self.global_mean_ = None
        
        # Validate strategy
        valid_strategies = ['naive_mean', 'feature_based']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
    
    def fit(self, X_train, y_train):
        """
        Learn baseline prediction values from training data.
        
        Args:
            X_train: Training features (DataFrame)
            y_train: Training target (Series)
            
        Returns:
            self (for method chaining)
        """

        self.global_mean_ = y_train.mean()

        if self.strategy == 'feature_based':
            if 'DESTINATION_AVG_DELAY' not in X_train.columns:
                raise ValueError("DESTINATION_AVG_DELAY feature not found in X_train!")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using fitted baseline.
        
        Args:
            X: Features (DataFrame)
            
        Returns:
            predictions: numpy array of predictions
        """
        # Strategy 1 - Naive Mean
        if self.strategy == 'naive_mean':

            return np.full(len(X), self.global_mean_)
        
        # Strategy 2 - Feature Based
        elif self.strategy == 'feature_based':

            if 'DESTINATION_AVG_DELAY' not in X.columns:
                raise ValueError("DESTINATION_AVG_DELAY feature not found!")
            
            predictions = X['DESTINATION_AVG_DELAY']
            
            # Handle any missing values (shouldn't happen, but just in case)
            predictions = predictions.fillna(self.global_mean_)  
            
            return predictions.values
    
    def evaluate(self, X, y_true):
        """
        Evaluate baseline performance.
        
        Args:
            X: Features (DataFrame)
            y_true: True target values (Series or array)
        
        Returns:
            Dictionary with RMSE, MAE, R²
        """
        # Get predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'strategy': self.strategy,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'n_samples': len(y_true)
        }
    
    def print_metrics(self, metrics):
        """
        Print formatted metrics.
        
        Args:
            metrics: Dictionary from evaluate()
        """
        print(f"\n{'='*60}")
        print(f"Heuristic Baseline: {metrics['strategy']}")
        print(f"{'='*60}")
        print(f"Samples:  {metrics['n_samples']:>10,}")
        print(f"RMSE:     {metrics['RMSE']:>10.2f} minutes")
        print(f"MAE:      {metrics['MAE']:>10.2f} minutes")
        print(f"R²:       {metrics['R2']:>10.4f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    print("="*60)
    print("TESTING HEURISTIC BASELINES")
    print("="*60)
    
    X_train = pd.read_csv("../../data/processed/X_train.csv")
    y_train = pd.read_csv("../../data/processed/y_train.csv")
    X_val  = pd.read_csv("../../data/processed/X_val.csv")
    y_val = pd.read_csv("../../data/processed/y_val.csv")
    
    strategies = ['naive_mean', 'feature_based']
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting: {strategy}")
        
        baseline = HeuristicBaseline(strategy=strategy)
        baseline.fit(X_train, y_train)
        
        metrics = baseline.evaluate(X_val, y_val)
        baseline.print_metrics(metrics)
        
        results[strategy] = metrics
    
    # Print comparison
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"{'-'*60}")
    
    for strategy, metrics in results.items():
        print(f"{strategy:<20} {metrics['RMSE']:>10.2f} {metrics['MAE']:>10.2f} {metrics['R2']:>10.4f}")
    
    print(f"{'='*60}")
    print("\n✅ Baseline evaluation complete!")
    print("\nNext step: Train Linear Regression and compare!")