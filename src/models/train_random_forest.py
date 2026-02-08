"""
Random Forest Model - Tree-Based Model.

This script:
1. Loads prepared modeling data (NO SCALING needed!)
2. Trains Random Forest Regressor
3. Evaluates performance vs Linear Regression
4. Analyzes feature importance (not coefficients)
5. Saves model for deployment
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


def load_data():
    """Load prepared modeling data."""
    print("Loading data...")

    X_train = pd.read_csv("../../data/processed/X_train.csv")
    y_train = pd.read_csv("../../data/processed/y_train.csv")
    X_val = pd.read_csv("../../data/processed/X_val.csv")
    y_val = pd.read_csv("../../data/processed/y_val.csv")
    
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    
    return X_train, y_train, X_val, y_val


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Regressor.
    
    Note: NO SCALING needed for tree-based models!
    """
    print("\nTraining Random Forest...")
    print("This may take a few minutes with 3.4M samples...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Model trained in {elapsed_time/60:.2f} minutes!")

    return model


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate model performance.
    
    Returns metrics dictionary.
    """
    print(f"\nEvaluating on {dataset_name}...")

    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        'dataset': dataset_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'n_samples': len(y)
    }


def print_metrics(metrics):
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"Random Forest - {metrics['dataset']}")
    print(f"{'='*60}")
    print(f"Samples:  {metrics['n_samples']:>10,}")
    print(f"RMSE:     {metrics['RMSE']:>10.2f} minutes")
    print(f"MAE:      {metrics['MAE']:>10.2f} minutes")
    print(f"R²:       {metrics['R2']:>10.4f}")
    print(f"{'='*60}")


def analyze_feature_importance(model, feature_names, top_n=15):
    """
    Analyze and display feature importance.
    
    Random Forest provides importance scores (not coefficients).
    Higher score = more important for prediction.
    """
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Random Forest)")
    print(f"{'='*60}")

    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Calculate cumulative importance
    importance_df['cumulative'] = importance_df['importance'].cumsum()
    
    # Display top features
    print(f"\nTop {top_n} Most Important Features:")
    print(f"{'Feature':<30} {'Importance':>12} {'Cumulative':>12}")
    print(f"{'-'*60}")
    
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"{row['feature']:<30} {row['importance']:>12.4f} {row['cumulative']:>12.2%}")
    
    # How many features explain 80% of importance?
    features_80 = (importance_df['cumulative'] <= 0.80).sum() + 1
    print(f"\n✅ Top {features_80} features explain 80% of total importance")
    
    return importance_df


def compare_with_linear_regression(rf_metrics):
    """Compare Random Forest with Linear Regression results."""
    print(f"\n{'='*60}")
    print("COMPARISON: Random Forest vs Linear Regression")
    print(f"{'='*60}")
    
    # Linear Regression results (from earlier)
    lr_rmse = 31.54
    lr_mae = 19.68
    lr_r2 = 0.0317
    
    # Calculate improvements
    rmse_improvement = (lr_rmse - rf_metrics['RMSE']) / lr_rmse * 100
    mae_improvement = (lr_mae - rf_metrics['MAE']) / lr_mae * 100
    r2_multiplier = lr_r2 / rf_metrics['R2']
    
    print(f"\n{'Metric':<15} {'Linear Reg':>12} {'Random Forest':>15} {'Improvement'}")
    print(f"{'-'*65}")
    print(f"{'RMSE (min)':<15} {lr_rmse:>12.2f} {rf_metrics['RMSE']:>15.2f} {rmse_improvement:>10.1f}%")
    print(f"{'MAE (min)':<15} {lr_mae:>12.2f} {rf_metrics['MAE']:>15.2f} {mae_improvement:>10.1f}%")
    print(f"{'R²':<15} {lr_r2:>12.4f} {rf_metrics['R2']:>15.4f} {r2_multiplier:>9.1f}x better")
    print(f"{'='*65}")


def save_model(model, output_dir='../../models'):
    """Save trained model for deployment."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(output_dir, "random_forest.pkl"))
    
    print(f"\n✅ Model saved: {output_dir}/random_forest.pkl")
    
    # Print model size
    model_size = os.path.getsize(f'{output_dir}/random_forest.pkl') / (1024 * 1024)
    print(f"   Model size: {model_size:.2f} MB")


def main():
    """Main execution function."""
    print("="*60)
    print("RANDOM FOREST - TREE-BASED MODEL")
    print("="*60)
    
    # Step 1: Load data (NO SCALING!)
    X_train, y_train, X_val, y_val = load_data()
    
    print("\n⚠️  Note: Tree-based models don't need feature scaling!")
    print("   Using raw features for training.")
    
    # Step 2: Train model
    model = train_random_forest(X_train, y_train)
    
    # Step 3: Evaluate on train set
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    print_metrics(train_metrics)
    
    # Step 4: Evaluate on validation set
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    print_metrics(val_metrics)
    
    # Step 5: Check for overfitting
    print(f"\n{'='*60}")
    print("OVERFITTING CHECK")
    print(f"{'='*60}")
    train_val_diff = train_metrics['RMSE'] - val_metrics['RMSE']
    print(f"Training RMSE:   {train_metrics['RMSE']:.2f}")
    print(f"Validation RMSE: {val_metrics['RMSE']:.2f}")
    print(f"Difference:      {train_val_diff:.2f} minutes")
    
    if abs(train_val_diff) < 1.0:
        print("✅ No significant overfitting detected!")
    elif train_metrics['RMSE'] < val_metrics['RMSE'] - 2:
        print("⚠️  Some overfitting detected (train much better than val)")
    else:
        print("✅ Model generalizes well!")
    
    # Step 6: Feature importance analysis
    importance_df = analyze_feature_importance(model, X_train.columns, top_n=15)
    
    # Step 7: Compare with Linear Regression
    compare_with_linear_regression(val_metrics)
    
    # Step 8: Save model
    save_model(model)
    
    print("\n" + "="*60)
    print("✅ RANDOM FOREST TRAINING COMPLETE!")
    print("="*60)
    
    # Optional: Save importance rankings for documentation
    importance_df.to_csv('../../models/rf_feature_importance.csv', index=False)
    print(f"\n✅ Feature importance saved: models/rf_feature_importance.csv")


if __name__ == "__main__":
    main()