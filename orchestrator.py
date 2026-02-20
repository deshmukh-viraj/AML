# orchestrator.py
"""
AML Experimentation Orchestrator

This script orchestrates the full experimentation pipeline:
1. Phase 1: Balancing Strategy Comparison
2. Phase 2: Model Comparison
3. Phase 3: Hyperparameter Tuning (for best model)

Results are logged to a single CSV file for analysis.

Usage:
    python orchestrator.py --data_path data/transactions.parquet
"""

import argparse
import gc
import os
import sys
import time
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.balancing import balance_data
from src.modeling import (
    train_and_evaluate,
    cross_validate_and_evaluate,
    MODEL_REGISTRY,
)


# =============================================================================
# Configuration
# =============================================================================

# Balancing methods to explore
BALANCING_METHODS = [
    "none",          # Baseline: no balancing
    "class_weight",  # Cost-sensitive learning
    "under_sample",  # Random under-sampling
    # "smote",       # Uncomment with caution - memory intensive
]

# Models to compare
MODELS = [
    "logistic_regression",
    "random_forest",
    # "lightgbm",    # Uncomment if lightgbm installed
]

# Hyperparameter grids for tuning phase
HYPERPARAMS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
    },
    # "lightgbm": {
    #     "num_leaves": [15, 31, 63],
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "n_estimators": [50, 100, 200],
    # },
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    path: str,
    target_col: str = "is_laundering",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Load data and perform train/test split.
    
    Args:
        path: Path to parquet file
        target_col: Name of target column
        test_size: Proportion for test split
        random_state: Reproducibility seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Memory optimization: convert to float32 for large datasets
    for col in X.select_dtypes(include=['float64']).columns:
        X[col] = X[col].astype(np.float32)
    
    # Simple random split (for time-series AML, consider time-based split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Loaded {len(df):,} transactions")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Positive rate (train): {y_train.mean():.4%}")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# Experiment Execution
# =============================================================================

def run_phase1_balancing_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results: List[Dict]
) -> None:
    """
    Phase 1: Compare balancing strategies with a baseline model.
    
    Uses Logistic Regression as fixed model to isolate balancing effect.
    """
    print("\n" + "="*60)
    print("PHASE 1: Balancing Strategy Comparison")
    print("="*60)
    
    baseline_model = "logistic_regression"
    model_params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
    
    for method in BALANCING_METHODS:
        print(f"\n--- Balancing: {method} ---")
        
        try:
            start_time = time.time()
            
            # Apply balancing
            X_balanced, y_balanced, sample_weights = balance_data(
                X_train, y_train, method=method, random_state=42
            )
            
            # Determine effective sample weights
            if method == "class_weight":
                effective_weights = sample_weights
            else:
                effective_weights = None
            
            # Train and evaluate
            result = train_and_evaluate(
                X_balanced, y_balanced,
                X_test, y_test,
                model_name=baseline_model,
                model_params=model_params,
                sample_weights=effective_weights,
            )
            
            elapsed = time.time() - start_time
            
            # Record results
            record = {
                "timestamp": datetime.now().isoformat(),
                "phase": 1,
                "balancing_method": method,
                "model_name": baseline_model,
                "params": json.dumps(model_params),
                "pr_auc": result["metrics"]["pr_auc"],
                "roc_auc": result["metrics"]["roc_auc"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1": result["metrics"]["f1"],
                "train_time_seconds": elapsed,
                "n_train_samples": len(X_balanced),
                "status": "success",
            }
            
            # Add operational metrics if available
            for key in result["metrics"]:
                if "recall_at_" in key:
                    record[key] = result["metrics"][key]
            
            results.append(record)
            print(f"  PR-AUC: {record['pr_auc']:.4f} | Recall: {record['recall']:.4f} | Time: {elapsed:.1f}s")
            
            # Memory cleanup
            del X_balanced, y_balanced
            gc.collect()
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "phase": 1,
                "balancing_method": method,
                "model_name": baseline_model,
                "status": f"error: {str(e)}",
            })


def run_phase2_model_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_balancing: str,
    results: List[Dict]
) -> str:
    """
    Phase 2: Compare models using best balancing strategy.
    
    Returns:
        Name of best performing model
    """
    print("\n" + "="*60)
    print("PHASE 2: Model Comparison")
    print("="*60)
    print(f"Using balancing method: {best_balancing}")
    
    best_model = None
    best_pr_auc = 0
    
    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")
        
        try:
            start_time = time.time()
            
            # Apply balancing
            X_balanced, y_balanced, sample_weights = balance_data(
                X_train, y_train, method=best_balancing, random_state=42
            )
            
            # Use default params for comparison
            model_params = {}
            
            # Train and evaluate
            result = train_and_evaluate(
                X_balanced, y_balanced,
                X_test, y_test,
                model_name=model_name,
                model_params=model_params,
                sample_weights=sample_weights,
            )
            
            elapsed = time.time() - start_time
            
            # Record results
            record = {
                "timestamp": datetime.now().isoformat(),
                "phase": 2,
                "balancing_method": best_balancing,
                "model_name": model_name,
                "params": json.dumps(model_params),
                "pr_auc": result["metrics"]["pr_auc"],
                "roc_auc": result["metrics"]["roc_auc"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1": result["metrics"]["f1"],
                "train_time_seconds": elapsed,
                "n_train_samples": len(X_balanced),
                "status": "success",
            }
            
            for key in result["metrics"]:
                if "recall_at_" in key:
                    record[key] = result["metrics"][key]
            
            results.append(record)
            print(f"  PR-AUC: {record['pr_auc']:.4f} | Recall: {record['recall']:.4f} | Time: {elapsed:.1f}s")
            
            # Track best model
            if result["metrics"]["pr_auc"] > best_pr_auc:
                best_pr_auc = result["metrics"]["pr_auc"]
                best_model = model_name
            
            del X_balanced, y_balanced
            gc.collect()
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "phase": 2,
                "balancing_method": best_balancing,
                "model_name": model_name,
                "status": f"error: {str(e)}",
            })
    
    if best_model is None:
        raise ValueError("No model succeeded in Phase 2")
    
    print(f"\nBest model: {best_model} (PR-AUC: {best_pr_auc:.4f})")
    return best_model


def run_phase3_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_balancing: str,
    best_model: str,
    results: List[Dict]
) -> Dict[str, Any]:
    """
    Phase 3: Lightweight hyperparameter tuning for best model.
    
    Uses random search with limited iterations.
    """
    print("\n" + "="*60)
    print("PHASE 3: Hyperparameter Tuning")
    print("="*60)
    print(f"Model: {best_model} | Balancing: {best_balancing}")
    
    if best_model not in HYPERPARAMS:
        print("No hyperparameter grid defined, skipping tuning")
        return {}
    
    param_grid = HYPERPARAMS[best_model]
    
    # Generate random search combinations
    param_combinations = []
    for _ in range(10):  # Limited iterations for exploration
        combo = {}
        for param, values in param_grid.items():
            combo[param] = np.random.choice(values)
        combo["random_state"] = 42
        param_combinations.append(combo)
    
    best_params = None
    best_pr_auc = 0
    
    for i, model_params in enumerate(param_combinations):
        print(f"\n--- Iteration {i+1}/{len(param_combinations)}: {model_params} ---")
        
        try:
            start_time = time.time()
            
            # Apply balancing
            X_balanced, y_balanced, sample_weights = balance_data(
                X_train, y_train, method=best_balancing, random_state=42
            )
            
            # Train and evaluate
            result = train_and_evaluate(
                X_balanced, y_balanced,
                X_test, y_test,
                model_name=best_model,
                model_params=model_params,
                sample_weights=sample_weights,
            )
            
            elapsed = time.time() - start_time
            
            pr_auc = result["metrics"]["pr_auc"]
            print(f"  PR-AUC: {pr_auc:.4f} | Time: {elapsed:.1f}s")
            
            # Record results
            record = {
                "timestamp": datetime.now().isoformat(),
                "phase": 3,
                "balancing_method": best_balancing,
                "model_name": best_model,
                "params": json.dumps(model_params),
                "pr_auc": pr_auc,
                "roc_auc": result["metrics"]["roc_auc"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1": result["metrics"]["f1"],
                "train_time_seconds": elapsed,
                "n_train_samples": len(X_balanced),
                "status": "success",
            }
            
            for key in result["metrics"]:
                if "recall_at_" in key:
                    record[key] = result["metrics"][key]
            
            results.append(record)
            
            # Track best
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_params = model_params
            
            del X_balanced, y_balanced
            gc.collect()
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "phase": 3,
                "balancing_method": best_balancing,
                "model_name": best_model,
                "params": json.dumps(model_params),
                "status": f"error: {str(e)}",
            })
    
    print(f"\nBest params: {best_params} (PR-AUC: {best_pr_auc:.4f})")
    return best_params or {}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AML Experimentation Orchestrator")
    parser.add_argument(
        "--data_path",
        type=str,
        default="input/transactions.parquet",
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/experiment_results.csv",
        help="Path to output results CSV"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="is_laundering",
        help="Target column name"
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_path, args.target_col)
    
    # Initialize results storage
    results = []
    
    # Phase 1: Balancing comparison
    run_phase1_balancing_experiments(
        X_train, y_train, X_test, y_test, results
    )
    
    # Determine best balancing (highest PR-AUC from Phase 1)
    phase1_results = [r for r in results if r.get("phase") == 1 and r.get("status") == "success"]
    if phase1_results:
        best_balancing = max(phase1_results, key=lambda x: x.get("pr_auc", 0))["balancing_method"]
    else:
        best_balancing = "none"
    
    # Phase 2: Model comparison
    best_model = run_phase2_model_comparison(
        X_train, y_train, X_test, y_test, best_balancing, results
    )
    
    # Phase 3: Hyperparameter tuning
    best_params = run_phase3_hyperparameter_tuning(
        X_train, y_train, X_test, y_test, best_balancing, best_model, results
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    print(f"\nResults saved to {args.output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Best balancing: {best_balancing}")
    print(f"Best model: {best_model}")
    print(f"Best params: {best_params}")
    
    # Show top results
    successful_results = [r for r in results if r.get("status") == "success"]
    if successful_results:
        print("\nTop 5 experiments by PR-AUC:")
        top5 = sorted(successful_results, key=lambda x: x.get("pr_auc", 0), reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            print(f"  {i}. {r['balancing_method']} + {r['model_name']}: PR-AUC={r['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
