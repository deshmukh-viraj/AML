# config.py
"""Configuration for AML experiments.

This file centralizes hyperparameters and experiment settings.
Edit this file to customize experiment runs without touching core code.
"""

# =============================================================================
# Balancing Strategy Configuration
# =============================================================================

# Balancing methods to explore in Phase 1
# Options: "none", "class_weight", "under_sample", "smote"
BALANCING_METHODS = [
    "none",
    "class_weight", 
    "under_sample",
    # "smote",  # Uncomment with caution - memory intensive for large datasets
]

# Under-sampling ratio (majority:minority)
# 1.0 = 1:1 balance, 2.0 = 2:1 majority:minority
UNDER_SAMPLE_RATIO = 1.0

# SMOTE settings (if enabled)
SMOTE_SAMPLING_STRATEGY = 1.0  # Target ratio
SMOTE_K_NEIGHBORS = 5


# =============================================================================
# Model Configuration
# =============================================================================

# Models to compare in Phase 2
MODELS = [
    "logistic_regression",
    "random_forest",
    # "lightgbm",  # Uncomment if lightgbm is installed
]


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
        "max_depth": [5, 10, 20, None],  # None = unlimited
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

# Memory optimization
USE_FLOAT32 = True  # Convert float64 to float32


# =============================================================================
# Evaluation Metrics
# =============================================================================

# Primary metric for model selection
PRIMARY_METRIC = "pr_auc"

# Precision thresholds for operational metrics
RECALL_AT_PRECISION_THRESHOLDS = [0.80, 0.90, 0.95]
