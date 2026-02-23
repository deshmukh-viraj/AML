"""
Balancing strategies for AML binary classification.

Responsibility:
    Transform (X_train, y_train) → (X_balanced, y_balanced, sample_weights)
    using one of four strategies: none, class_weight, under_sample, smote.

Design decisions:
    - Index-based under-sampling: never copies the full DataFrame.
    - Returns a uniform (X, y, weights | None) tuple so the orchestrator
      needs no method-specific logic.
    - SMOTE is gated behind an explicit import check and a loud warning,
      because it generates synthetic transactions unsuitable for production.
    - All randomness is seeded for reproducibility.

Dependencies: numpy, pandas, (optional) imbalanced-learn>=0.11
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


BalanceResult = Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]


# Internal helpers

def _class_weights(y: pd.Series) -> dict:
    """Inverse-frequency weights: {class_label: weight}."""
    counts = y.value_counts()
    total = len(y)
    n_classes = len(counts)
    return {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}


def _sample_weights_from_class_weights(y: pd.Series) -> np.ndarray:
    """Map each sample to its class weight."""
    cw = _class_weights(y)
    return np.array([cw[label] for label in y], dtype=np.float32)


def _random_under_sample(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Under-sample majority class to majority:minority = ratio:1.

    Uses index arithmetic — no DataFrame copy until the final .loc[].
    """
    rng = np.random.default_rng(random_state)

    minority_idx = y.index[y == 1].tolist()
    majority_idx = y.index[y == 0].tolist()

    n_keep = int(len(minority_idx) * ratio)
    n_keep = min(n_keep, len(majority_idx))  # Can't keep more than available

    sampled_majority = rng.choice(majority_idx, size=n_keep, replace=False)
    selected = np.sort(np.concatenate([minority_idx, sampled_majority]))

    logger.info(
        "Under-sampling: minority=%d, majority kept=%d (ratio=%.1f)",
        len(minority_idx), n_keep, ratio,
    )
    return X.loc[selected], y.loc[selected]


def _smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float,
    random_state: int,
    k_neighbors: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE over-sampling via imbalanced-learn.

    WARNING: Synthetic transactions may not reflect real laundering patterns.
    Use only for offline experimentation, never in a production scoring path.
    """
    warnings.warn(
        "SMOTE creates synthetic AML transactions that may not reflect real "
        "laundering behaviour. Use for exploration only — not production.",
        UserWarning,
        stacklevel=3,
    )

    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE.  "
            "Install it with: pip install imbalanced-learn"
        ) from exc

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    X_arr, y_arr = smote.fit_resample(X.values.astype(np.float32), y.values)

    logger.info("SMOTE: %d → %d samples", len(y), len(y_arr))

    new_idx = range(len(y_arr))
    return (
        pd.DataFrame(X_arr, columns=X.columns, index=new_idx),
        pd.Series(y_arr, name=y.name, index=new_idx),
    )



VALID_METHODS = frozenset({"none", "class_weight", "under_sample", "smote"})


def balance_data(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "none",
    *,
    random_state: int = 42,
    under_sample_ratio: float = 1.0,
    smote_sampling_strategy: float = 1.0,
    smote_k_neighbors: int = 5,
) -> BalanceResult:
    """
    Unified entry-point for class-imbalance handling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (will not be mutated).
    y : pd.Series
        Binary target (0 = legitimate, 1 = suspicious).
    method : str
        One of ``"none"``, ``"class_weight"``, ``"under_sample"``, ``"smote"``.
    random_state : int
        Seed for reproducibility.
    under_sample_ratio : float
        Majority-to-minority ratio for under-sampling (1.0 = balanced 1:1).
    smote_sampling_strategy : float
        Target minority fraction for SMOTE.
    smote_k_neighbors : int
        k for SMOTE interpolation.

    Returns
    -------
    (X_out, y_out, sample_weights)
        ``sample_weights`` is an ndarray only for ``"class_weight"``; else ``None``.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown balancing method {method!r}. Choose from {sorted(VALID_METHODS)}."
        )

    pos_rate = y.mean()
    logger.info("balance_data: method=%s, n=%d, positive_rate=%.4f", method, len(y), pos_rate)

    if method == "none":
        return X, y, None

    if method == "class_weight":
        weights = _sample_weights_from_class_weights(y)
        logger.info("class_weight: weights=%s", _class_weights(y))
        return X, y, weights

    if method == "under_sample":
        X_bal, y_bal = _random_under_sample(X, y, under_sample_ratio, random_state)
        return X_bal, y_bal, None

    # method == "smote"
    X_bal, y_bal = _smote(X, y, smote_sampling_strategy, random_state, smote_k_neighbors)
    return X_bal, y_bal, None