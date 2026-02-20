# src/modeling.py
"""Model training and evaluation utilities for AML classification.

This module provides lightweight wrappers for classification models
suitable for imbalanced AML datasets.

Key design decisions:
- Unified interface accepting sample_weights for cost-sensitive learning
- AML-specific metrics (PR-AUC prioritized over ROC-AUC)
- Minimal dependencies: sklearn base + optional LightGBM
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)


# =============================================================================
# Metrics
# =============================================================================

def compute_aml_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute AML-appropriate evaluation metrics.
    
    Focus on PR-AUC because:
    - ROC-AUC is misleading with extreme imbalance (TN dominates)
    - PR-AUC reflects model performance on the rare positive class
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}
    
    # Basic classification metrics
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    
    # PR-AUC (primary metric for AML)
    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        
        # ROC-AUC (secondary, for reference)
        # Only compute if both classes present
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        else:
            metrics["roc_auc"] = np.nan
        
        # Recall at specific precision thresholds (operational metrics)
        # These answer: "If we need X% precision, what recall can we achieve?"
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        
        for target_precision in [0.80, 0.90, 0.95]:
            # Find recall where precision >= target
            mask = precision_vals >= target_precision
            if mask.any():
                metrics[f"recall_at_{target_precision:.0%}_precision"] = float(recall_vals[mask].max())
            else:
                metrics[f"recall_at_{target_precision:.0%}_precision"] = 0.0
    
    return metrics


# =============================================================================
# Model Definitions
# =============================================================================

def create_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> LogisticRegression:
    """
    Create Logistic Regression model.
    
    Args:
        C: Inverse regularization strength
        max_iter: Maximum iterations for convergence
        random_state: Reproducibility seed
        
    Returns:
        Fitted LogisticRegression instance
    """
    return LogisticRegression(
        C=C,
        solver="saga",  # Supports l1/l2/elasticnet
        penalty="l2",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )


def create_random_forest(
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    **kwargs
) -> RandomForestClassifier:
    """
    Create Random Forest model.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None for unlimited)
        random_state: Reproducibility seed
        
    Returns:
        Fitted RandomForestClassifier instance
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"  # Built-in handling for imbalance
    )


def create_lightgbm(
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42,
    **kwargs
):
    """
    Create LightGBM model (requires lightgbm package).
    
    Args:
        num_leaves: Maximum leaves per tree
        learning_rate: Step size shrinkage
        n_estimators: Number of boosting iterations
        random_state: Reproducibility seed
        
    Returns:
        Fitted LGBMClassifier instance
        
    Raises:
        ImportError: If lightgbm not installed
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = kwargs.get("scale_pos_weight", None)
    
    return lgb.LGBMClassifier(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1
    )


MODEL_REGISTRY = {
    "logistic_regression": create_logistic_regression,
    "random_forest": create_random_forest,
    "lightgbm": create_lightgbm,
}


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    model_params: Optional[Dict[str, Any]] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Train model and compute evaluation metrics.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of model from registry
        model_params: Hyperparameters for model
        sample_weights: Optional sample weights for cost-sensitive learning
        
    Returns:
        Dictionary containing metrics, model, and metadata
    """
    if model_params is None:
        model_params = {}
    
    random_state = model_params.get("random_state", 42)
    
    # Create model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_fn = MODEL_REGISTRY[model_name]
    model = model_fn(**model_params)
    
    # Train with timing
    start_time = time.time()
    
    # Handle sample weights based on model type
    fit_kwargs = {}
    if sample_weights is not None and model_name == "logistic_regression":
        fit_kwargs["sample_weight"] = sample_weights
    
    model.fit(X_train, y_train, **fit_kwargs)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = compute_aml_metrics(y_test, y_pred, y_proba)
    metrics["train_time_seconds"] = train_time
    
    return {
        "model": model,
        "metrics": metrics,
        "model_name": model_name,
        "model_params": model_params,
        "balancing_method": "unknown",  # Set by orchestrator
    }


def cross_validate_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_params: Optional[Dict[str, Any]] = None,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Stratified K-Fold cross-validation with metric aggregation.
    
    Args:
        X: Full feature set
        y: Full labels
        model_name: Model from registry
        model_params: Hyperparameters
        sample_weights: Optional sample weights
        n_splits: Number of CV folds
        random_state: Reproducibility seed
        
    Returns:
        Dictionary with aggregated metrics (mean ± std)
    """
    if model_params is None:
        model_params = {}
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Handle weights for this fold
        fold_weights = None
        if sample_weights is not None:
            fold_weights = sample_weights[train_idx]
        
        result = train_and_evaluate(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            model_name=model_name,
            model_params=model_params,
            sample_weights=fold_weights,
        )
        
        fold_metrics.append(result["metrics"])
    
    # Aggregate metrics
    aggregated = {}
    metric_names = fold_metrics[0].keys()
    
    for metric in metric_names:
        values = [fm[metric] for fm in fold_metrics if not np.isnan(fm.get(metric, np.nan))]
        if values:
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
    
    return {
        "model_name": model_name,
        "model_params": model_params,
        "fold_metrics": fold_metrics,
        "aggregated_metrics": aggregated,
    }
