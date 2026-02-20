"""Balancing strategies for AML classification.

This module provides lightweight implementations of class imbalance techniques
suitable for exploration with large financial transaction datasets.

Key design decisions:
- Memory-aware: prefer index-based slicing over full DataFrame copies
- Returns (X, y, sample_weights) tuple for uniform interface
- No in-place mutations of input data
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


def compute_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Compute inverse frequency class weights for binary classification.
    
    For AML: rare positive class (laundering) gets higher weight.
    
    Args:
        y: Binary target series (0=legitimate, 1=suspicious)
        
    Returns:
        Dictionary mapping class to weight
    """
    n_samples = len(y)
    classes, counts = np.unique(y, return_counts=True)
    weights = n_samples / (len(classes) * counts)
    return dict(zip(classes, weights))


def apply_class_weights(y: pd.Series) -> np.ndarray:
    """
    Apply computed class weights to each sample.
    
    Args:
        y: Binary target series
        
    Returns:
        Array of sample weights matching y length
    """
    weights = compute_class_weights(y)
    return np.array([weights[label] for label in y])


def random_under_sample(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Random under-sampling of majority class.
    
    Memory-efficient: uses index-based selection rather than copying DataFrame.
    
    Args:
        X: Feature DataFrame
        y: Target series
        ratio: Ratio of majority to minority samples (1.0 = 1:1 balance)
        random_state: Reproducibility seed
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    np.random.seed(random_state)
    
    minority_mask = y == 1
    majority_mask = y == 0
    
    minority_idx = y[minority_mask].index.tolist()
    majority_idx = y[majority_mask].index.tolist()
    
    n_minority = len(minority_idx)
    n_majority_to_keep = int(n_minority * ratio)
    
    sampled_majority = np.random.choice(majority_idx, size=n_majority_to_keep, replace=False)
    selected_idx = np.concatenate([minority_idx, sampled_majority])
    
    # Preserve original order for reproducibility
    selected_idx = np.sort(selected_idx)
    
    return X.loc[selected_idx], y.loc[selected_idx]


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 1.0,
    random_state: int = 42,
    k_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE-style synthetic oversampling (requires imbalanced-learn).
    
    WARNING: SMOTE has known issues for AML:
    - May generate unrealistic synthetic fraud patterns
    - Memory-intensive for high-dimensional data
    - Creates data that never existed in production
    
    Use only for exploration, never in production.
    
    Args:
        X: Feature DataFrame
        y: Target series
        sampling_strategy: Target ratio of minority to majority
        random_state: Reproducibility seed
        k_neighbors: Neighbors for interpolation
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    warnings.warn(
        "SMOTE generates synthetic transactions that may not reflect "
        "real laundering patterns. Use only for exploration, not production.",
        UserWarning
    )
    
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn required for SMOTE. Install with: pip install imbalanced-learn"
        )
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors
    )
    
    X_array = X.values.astype(np.float32)
    X_resampled, y_resampled = smote.fit_resample(X_array, y)
    
    # Reconstruct DataFrame with original index structure
    new_index = range(len(y_resampled))
    X_result = pd.DataFrame(X_resampled, columns=X.columns, index=new_index)
    y_result = pd.Series(y_resampled, name=y.name, index=new_index)
    
    return X_result, y_result


def balance_data(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "none",
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """
    Unified interface for all balancing methods.
    
    Args:
        X: Feature DataFrame
        y: Target series
        method: Balancing strategy ('none', 'class_weight', 'under_sample', 'smote')
        **kwargs: Additional arguments passed to specific methods
        
    Returns:
        Tuple of (X_transformed, y_transformed, sample_weights)
        - sample_weights is None unless method='class_weight'
    """
    random_state = kwargs.get("random_state", 42)
    
    if method == "none":
        # Baseline: no transformation
        return X, y, None
    
    elif method == "class_weight":
        # Cost-sensitive learning: compute weights without resampling
        # Memory-efficient: no data duplication
        weights = apply_class_weights(y)
        return X, y, weights
    
    elif method == "under_sample":
        # Reduce majority class to match minority
        ratio = kwargs.get("ratio", 1.0)
        X_balanced, y_balanced = random_under_sample(
            X, y, ratio=ratio, random_state=random_state
        )
        return X_balanced, y_balanced, None
    
    elif method == "smote":
        # Synthetic oversampling (with warnings)
        X_balanced, y_balanced = apply_smote(
            X, y,
            sampling_strategy=kwargs.get("sampling_strategy", 1.0),
            random_state=random_state,
            k_neighbors=kwargs.get("k_neighbors", 5)
        )
        return X_balanced, y_balanced, None
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
