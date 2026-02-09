"""
Linear Regression Model - First ML Model.

This script:
1. Loads prepared modeling data
2. Standardizes features using StandardScaler
3. Trains Linear Regression model
4. Evaluates performance vs baseline
5. Analyzes feature coefficients for interpretability
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn


def load_data():
    """Load prepared modeling data."""
    print("Loading data...")
    
    # Load X and y for train and validation
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv")
    
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    
    return X_train, y_train, X_val, y_val


def standardize_features(X_train, X_val):
    """
    Standardize features to mean=0, std=1.
    
    Important: Fit scaler ONLY on training data!
    """
    print("\nStandardizing features...")
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert back to DataFrame (keep column names)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    
    print("✅ Features standardized (mean=0, std=1)")
    
    return X_train_scaled, X_val_scaled, scaler


def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""
    print("\nTraining Linear Regression...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("✅ Model trained!")

    return model


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate model performance.
    
    Returns metrics dictionary.
    """
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
    print(f"Linear Regression - {metrics['dataset']}")
    print(f"{'='*60}")
    print(f"Samples:  {metrics['n_samples']:>10,}")
    print(f"RMSE:     {metrics['RMSE']:>10.2f} minutes")
    print(f"MAE:      {metrics['MAE']:>10.2f} minutes")
    print(f"R²:       {metrics['R2']:>10.4f}")
    print(f"{'='*60}")


def analyze_coefficients(model, feature_names, top_n=10):
    """
    Analyze and display top feature coefficients.
    
    Shows which features have strongest positive/negative impact.
    """
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Top Coefficients)")
    print(f"{'='*60}")
    
    coefficients = model.coef_.ravel()   
    feature_names_list = list(feature_names)  
    
    # Create DataFrame for easy sorting
    coef_df = pd.DataFrame({
        'feature': feature_names_list,
        'coefficient': coefficients
    })

    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    # Display top positive and negative
    print(f"\nTop {top_n} Most Important Features (by magnitude):")
    print(f"{'Feature':<30} {'Coefficient':>12} {'Impact'}")
    print(f"{'-'*60}")
    
    for idx, row in coef_df.head(top_n).iterrows():
        impact = "↑ increases delay" if row['coefficient'] > 0 else "↓ decreases delay"
        print(f"{row['feature']:<30} {row['coefficient']:>12.4f}  {impact}")
    
    print(f"\nIntercept: {model.intercept_[0]:.4f}")
    
    return coef_df


def compare_with_baseline(lr_metrics):
    """Compare Linear Regression with baseline results."""
    print(f"\n{'='*60}")
    print("COMPARISON: Linear Regression vs Baseline")
    print(f"{'='*60}")
    
    # Baseline results (from earlier)
    baseline_rmse = 31.96  # feature-based baseline
    baseline_mae = 20.00
    baseline_r2 = 0.0059

    rmse_improvement = (baseline_rmse - lr_metrics['RMSE']) / baseline_rmse * 100
    mae_improvement = (baseline_mae - lr_metrics['MAE']) / baseline_mae * 100
    r2_multiplier = baseline_r2 / lr_metrics['R2']
    
    print(f"\n{'Metric':<15} {'Baseline':>12} {'Linear Reg':>12} {'Improvement'}")
    print(f"{'-'*60}")
    print(f"{'RMSE (min)':<15} {baseline_rmse:>12.2f} {lr_metrics['RMSE']:>12.2f} {rmse_improvement:>10.1f}%")
    print(f"{'MAE (min)':<15} {baseline_mae:>12.2f} {lr_metrics['MAE']:>12.2f} {mae_improvement:>10.1f}%")
    print(f"{'R²':<15} {baseline_r2:>12.4f} {lr_metrics['R2']:>12.4f} {r2_multiplier:>9.1f}x better")
    print(f"{'='*60}")


def save_model_and_scaler(model, scaler, output_dir='models'):
    """Save trained model and scaler for deployment."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, os.path.join(output_dir, "linear_regression.pkl"))    
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    
    print(f"\n✅ Model saved: {output_dir}/linear_regression.pkl")
    print(f"✅ Scaler saved: {output_dir}/scaler.pkl")


def main():
    """Main execution function."""
    print("="*60)
    print("LINEAR REGRESSION - FIRST ML MODEL")
    print("="*60)
    
    mlflow.set_experiment("flight-delay-prediction")

    # Step 1: Load data
    X_train, y_train, X_val, y_val = load_data()
    
    # Step 2: Standardize features
    X_train_scaled, X_val_scaled, scaler = standardize_features(X_train, X_val)
    
    # Step 3: Train model
    model = train_linear_regression(X_train_scaled, y_train)
    
    # Step 4: Evaluate on train set (sanity check)
    train_metrics = evaluate_model(model, X_train_scaled, y_train, "Training")
    print_metrics(train_metrics)
    
    # Step 5: Evaluate on validation set (real performance)
    val_metrics = evaluate_model(model, X_val_scaled, y_val, "Validation")
    print_metrics(val_metrics)
    
    # Step 6: Analyze coefficients
    coef_df = analyze_coefficients(model, X_train.columns, top_n=15)
    
    # Step 7: Compare with baseline
    compare_with_baseline(val_metrics)

    with mlflow.start_run(run_name=f"linear_regression"):
            # Log parameters
            mlflow.log_param("model_type", "linear_regression")
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))      
            # Log metrics
            mlflow.log_metric("val_rmse", val_metrics['RMSE'])
            mlflow.log_metric("val_mae", val_metrics['MAE'])
            mlflow.log_metric("val_r2", val_metrics['R2'])
            
            # Log tags
            mlflow.set_tag("model_category", "linear_regression")
    
    # Step 8: Save model
    save_model_and_scaler(model, scaler)
    
    print("\n" + "="*60)
    print("✅ LINEAR REGRESSION TRAINING COMPLETE!")
    print("="*60)
    print("\n✅ All runs logged to MLflow!")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5000")

if __name__ == "__main__":
    main()