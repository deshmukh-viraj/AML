# config.py
"""Configuration for AML experiments.

This file centralizes hyperparameters and experiment settings.
Edit this file to customize experiment runs without touching core code.

Memory-Safe Defaults:
- USE_POLARS: True (uses Arrow memory format for efficiency)
- STREAMING_MODE: False (set to True for very large files > 10GB)
- FORCE_FLOAT32: True (reduces memory by 50%)
- CHUNK_SIZE: 100000 (balance between memory and speed)
"""

from typing import List

# =============================================================================
# Data Loading Configuration
# =============================================================================

# Use Polars for memory-efficient parquet loading (recommended)
# Polars uses Arrow memory format which is more efficient than pandas
USE_POLARS = True

# Use streaming mode for very large files (> 10GB)
# Trades some speed for lower memory usage
STREAMING_MODE = False

# Force float32 conversion for memory efficiency
# Reduces memory usage by ~50% for float64 columns
FORCE_FLOAT32 = True

# Chunk size for chunked processing (balancing, incremental training)
CHUNK_SIZE = 100000


# =============================================================================
# Balancing Strategy Configuration
# =============================================================================

# Balancing methods to explore in Phase 1
# Options: "none", "class_weight", "under_sample", "smote", "smoteenn", "adasyn"
BALANCING_METHODS = [
    "none",
    "class_weight", 
    "under_sample",
    "smoteenn",     # NEW: SMOTE + ENN cleaning (recommended)
    "adasyn",       # NEW: Adaptive synthetic sampling
    # "smote",      # Keep commented - memory intensive for large datasets
]

# Active balancing method for production use
# Options: "none", "class_weight", "under_sample", "smoteenn", "adasyn"
ACTIVE_BALANCING_METHOD = "smoteenn"

# Under-sampling ratio (majority:minority)
# 1.0 = 1:1 balance, 2.0 = 2:1 majority:minority
UNDER_SAMPLE_RATIO = 1.0

# SMOTE settings (if enabled)
SMOTE_SAMPLING_STRATEGY = 1.0
SMOTE_K_NEIGHBORS = 5

# SMOTE-ENN settings
SMOTEENN_SAMPLING_STRATEGY = "auto"  # "auto" or float ratio
SMOTEENN_SMOTE_K_NEIGHBORS = 5
SMOTEENN_ENN_N_NEIGHBORS = 3  # ENN cleans noisy samples

# ADASYN settings
ADASYN_SAMPLING_STRATEGY = "auto"  # "auto" or float ratio
ADASYN_N_NEIGHBORS = 5
ADASYN_RANDOM_STATE = 42


# =============================================================================
# Balanced Random Forest Configuration
# =============================================================================

# Use BalancedRandomForest (handles imbalance internally during training)
USE_BALANCED_RANDOM_FOREST = True

BALANCED_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "sampling_strategy": "all",  # Balance all classes
    "replacement": True,  # Bootstrap with replacement
    "n_jobs": -1
}


# =============================================================================
# Focal Loss Configuration (for deep learning)
# =============================================================================

# Focal Loss parameters (for PyTorch/TensorFlow)
FOCAL_LOSS_ALPHA = 0.25  # Class weight for minority class
FOCAL_LOSS_GAMMA = 2.0   # Focus parameter (higher = more focus on hard examples)


# =============================================================================
# Memory Management
# =============================================================================

# Memory safety settings
ENABLE_MEMORY_CHECKS = True
MAX_MEMORY_GB = 8  # Maximum memory to use before warning

# Memory thresholds for automatic strategies
LARGE_DATASET_ROWS = 1_000_000  # Rows threshold for chunked processing
VERY_LARGE_DATASET_ROWS = 10_000_000  # Rows threshold for streaming mode


# =============================================================================
# Model Configuration
# =============================================================================

# Models to compare in Phase 2
MODELS = [
    "logistic_regression",
    "random_forest",
    "balanced_random_forest",  # NEW: Built-in balancing
    # "lightgbm",  # Uncomment if lightgbm is installed
]

# Enable incremental learning for large datasets
# Uses SGDClassifier with partial_fit for memory efficiency
USE_INCREMENTAL_LEARNING = False  # Set to True for datasets > 10M rows


# =============================================================================
# Hyperparameter Grids
# =============================================================================

# Phase 3: Hyperparameter search space
# Each model maps to a dict of param -> list of values to try
HYPERPARAMS = {
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000],
        "solver": ["saga"],  # Supports L1, L2, elasticnet
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, 20, None],  # None = unlimited
        "min_samples_split": [2, 5, 10],
    },
    "balanced_random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    # "lightgbm": {
    #     "num_leaves": [15, 31, 63],
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "n_estimators": [50, 100, 200],
    #     "subsample": [0.8, 1.0],
    #     "colsample_bytree": [0.8, 1.0],
    # },
}


# =============================================================================
# Training Configuration
# =============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Force deterministic operations where possible
DETERMINISTIC = True

# Train/test split ratio
TEST_SIZE = 0.2

# Cross-validation settings (for Phase 3 if using CV)
N_CV_SPLITS = 5
N_HYPERPARAM_ITERATIONS = 10  # Random search iterations


# =============================================================================
# Data Configuration
# =============================================================================

# Target column name
TARGET_COLUMN = "is_laundering"

# Feature columns (if None, all columns except target are used)
# FEATURE_COLUMNS = None
FEATURE_COLUMNS = [
    # Add your feature column names here
    # "transaction_amount",
    # "transaction_frequency_7d",
    # "account_age_days",
    # ...
]


# =============================================================================
# Evaluation Metrics
# =============================================================================

# Primary metric for model selection
PRIMARY_METRIC = "pr_auc"

# Precision thresholds for operational metrics
RECALL_AT_PRECISION_THRESHOLDS = [0.80, 0.90, 0.95]


# =============================================================================
# Validation Settings
# =============================================================================

# Validation strategy
VALIDATION_STRATEGY = "stratified"  # "stratified" or "time_based"

# Minimum acceptable recall at precision threshold
MIN_RECALL_AT_80_PRECISION = 0.15  # Minimum fraud detection rate


# =============================================================================
# Memory-Safe Helper Functions
# =============================================================================

def get_memory_safe_settings(n_rows: int, file_size_gb: float) -> dict:
    """
    Determine memory-safe settings based on dataset size.
    
    Args:
        n_rows: Number of rows in the dataset
        file_size_gb: Size of the parquet file in GB
        
    Returns:
        Dictionary of recommended settings
    """
    settings = {
        "use_polars": USE_POLARS,
        "streaming": STREAMING_MODE,
        "force_float32": FORCE_FLOAT32,
        "chunk_size": CHUNK_SIZE,
    }
    
    # Adjust settings based on dataset size
    if n_rows > VERY_LARGE_DATASET_ROWS or file_size_gb > 10:
        # Very large dataset - enable all memory optimizations
        settings["streaming"] = True
        settings["chunk_size"] = CHUNK_SIZE // 2  # Smaller chunks
        print(f"Very large dataset detected ({n_rows:,} rows, {file_size_gb:.1f}GB)")
        print("  Enabling streaming mode and reducing chunk size")
        
    elif n_rows > LARGE_DATASET_ROWS:
        # Large dataset - enable Polars and float32
        settings["chunk_size"] = CHUNK_SIZE
        print(f"Large dataset detected ({n_rows:,} rows)")
        
    return settings


def validate_config() -> List[str]:
    """
    Validate configuration and return list of warnings.
    
    Returns:
        List of warning messages
    """
    warnings = []
    
    if USE_POLARS:
        try:
            import polars as pl
        except ImportError:
            warnings.append("USE_POLARS=True but Polars not installed. Falling back to pandas.")
    
    if STREAMING_MODE and not USE_POLARS:
        warnings.append("STREAMING_MODE=True but USE_POLARS=False. Streaming requires Polars.")
    
    return warnings
