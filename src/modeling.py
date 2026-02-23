# src/modeling.py
"""
Model training and evaluation for AML classification.

Responsibility:
    - Instantiate sklearn / LightGBM classifiers through a registry.
    - Train with optional sample weights (cost-sensitive learning).
    - Compute AML-appropriate metrics (PR-AUC primary, ROC-AUC secondary).
    - Provide a stratified cross-validation helper.

Design decisions:
    - PR-AUC is the primary metric because ROC-AUC is misleading under
      extreme class imbalance (the huge true-negative pool inflates the score).
    - Metrics include "recall at X% precision" thresholds — the operational
      question an AML team actually asks: "if we need 90% precision, how many
      real laundering cases do we catch?"
    - Logistic Regression uses the SAGA solver so L1, L2 and elastic-net
      penalties are all available without changing the factory function.
    - LightGBM is imported lazily so the module loads cleanly without it.

Dependencies: numpy, pandas, scikit-learn>=1.3, (optional) lightgbm>=4.0
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Precision thresholds that matter operationally.
_PRECISION_THRESHOLDS: Tuple[float, ...] = (0.80, 0.90, 0.95)


def compute_aml_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute AML-appropriate classification metrics.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth binary labels.
    y_pred : array-like of int
        Hard predictions.
    y_proba : array-like of float, optional
        Predicted probability for the positive class.
        Required for PR-AUC and ROC-AUC.

    Returns
    -------
    dict
        Keys: precision, recall, f1, pr_auc, roc_auc, confusion-matrix
        components, and recall_at_{P}%_precision for each threshold.
    """
    metrics: Dict[str, float] = {}

    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
    )

    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)

        n_pos = int(y_true.sum())
        if n_pos > 0 and n_pos < len(y_true):
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        else:
            metrics["roc_auc"] = float("nan")
            logger.warning("roc_auc skipped: only one class present in y_true.")

        prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_proba)
        for thr in _PRECISION_THRESHOLDS:
            mask = prec_vals >= thr
            key = f"recall_at_{thr:.0%}_precision"
            metrics[key] = float(rec_vals[mask].max()) if mask.any() else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    **_,
) -> LogisticRegression:
    """L2-penalised logistic regression with the SAGA solver."""
    return LogisticRegression(
        C=C,
        solver="saga",
        penalty="l2",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )


def _make_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 2,
    random_state: int = 42,
    **_,
) -> RandomForestClassifier:
    """Random Forest with built-in balanced class weighting."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",  # handles imbalance without resampling
        random_state=random_state,
        n_jobs=-1,
    )


def _make_lightgbm(**kwargs):
    """LightGBM classifier (requires lightgbm package)."""
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Run: pip install lightgbm"
        ) from exc

    defaults = dict(
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(kwargs)
    return lgb.LGBMClassifier(**defaults)


# Registry maps string keys → factory functions.
MODEL_REGISTRY: Dict[str, Any] = {
    "logistic_regression": _make_logistic_regression,
    "random_forest": _make_random_forest,
    "lightgbm": _make_lightgbm,
}


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

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
    Train one model and evaluate it on a held-out test set.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : evaluation data (never seen during training)
    model_name : str
        Key in ``MODEL_REGISTRY``.
    model_params : dict, optional
        Passed verbatim to the model factory.
    sample_weights : np.ndarray, optional
        Per-sample weights for the training step (cost-sensitive learning).
        Only applied when the underlying estimator's ``fit`` method accepts
        a ``sample_weight`` argument.

    Returns
    -------
    dict with keys: ``model``, ``metrics``, ``model_name``, ``model_params``.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )

    params = dict(model_params or {})
    model = MODEL_REGISTRY[model_name](**params)

    # Determine whether this estimator accepts sample_weight in .fit().
    import inspect
    fit_sig = inspect.signature(model.fit)
    fit_kwargs: Dict[str, Any] = {}
    if sample_weights is not None and "sample_weight" in fit_sig.parameters:
        fit_kwargs["sample_weight"] = sample_weights
    elif sample_weights is not None:
        logger.warning(
            "%s.fit() does not accept sample_weight; weights ignored.", model_name
        )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_aml_metrics(
        y_test.values, y_pred, y_proba
    )
    metrics["train_time_seconds"] = round(elapsed, 3)

    logger.info(
        "train_and_evaluate: model=%s PR-AUC=%.4f Recall=%.4f Time=%.1fs",
        model_name, metrics.get("pr_auc", float("nan")),
        metrics["recall"], elapsed,
    )

    return {
        "model": model,
        "metrics": metrics,
        "model_name": model_name,
        "model_params": params,
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
    Stratified K-Fold cross-validation with metric aggregation (mean ± std).

    Parameters
    ----------
    X, y : full dataset (splitting is done internally)
    n_splits : number of CV folds
    All other parameters: same as ``train_and_evaluate``.

    Returns
    -------
    dict with keys: ``model_name``, ``model_params``, ``fold_metrics``,
    ``aggregated_metrics`` (each metric as ``{name}_mean`` / ``{name}_std``).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info("CV fold %d / %d", fold_idx + 1, n_splits)

        fold_weights = (
            sample_weights[train_idx] if sample_weights is not None else None
        )
        result = train_and_evaluate(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx], y.iloc[val_idx],
            model_name=model_name,
            model_params=model_params,
            sample_weights=fold_weights,
        )
        fold_metrics.append(result["metrics"])

    # Aggregate — skip NaN values silently.
    all_keys = fold_metrics[0].keys()
    aggregated: Dict[str, float] = {}
    for key in all_keys:
        vals = [
            fm[key] for fm in fold_metrics
            if not np.isnan(float(fm.get(key, float("nan"))))
        ]
        if vals:
            aggregated[f"{key}_mean"] = float(np.mean(vals))
            aggregated[f"{key}_std"] = float(np.std(vals))

    return {
        "model_name": model_name,
        "model_params": dict(model_params or {}),
        "fold_metrics": fold_metrics,
        "aggregated_metrics": aggregated,
    }