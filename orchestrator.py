# orchestrator.py
"""
AML Experimentation Orchestrator
=================================

Runs a three-phase experiment pipeline:

    Phase 1 — Balancing strategy comparison
        Fix model = Logistic Regression, vary balancing method.
        Goal: identify which imbalance strategy maximises PR-AUC.

    Phase 2 — Model comparison
        Fix balancing = best from Phase 1, vary model family.
        Goal: identify the best base learner.

    Phase 3 — Hyperparameter tuning (random search)
        Fix model + balancing from Phases 1–2, search param grid.
        Goal: squeeze out the last few PR-AUC points.

All results land in a single CSV so every experiment is reproducible.

Usage:
    python orchestrator.py --data_path data/transactions.parquet

Memory contract:
    Each phase discards balanced DataFrames immediately after the run.
    gc.collect() is called after each iteration to return RAM promptly.
    Suitable for 8 GB machines with datasets up to ~5 M rows (depending
    on feature count).

Dependencies: numpy, pandas, scikit-learn
    (+ lightgbm and/or imbalanced-learn if those methods are enabled)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.balancing import balance_data
from src.modeling import MODEL_REGISTRY, train_and_evaluate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration
# Edit these lists to add or remove methods / models / param grids.
# ---------------------------------------------------------------------------

BALANCING_METHODS: List[str] = [
    "none",
    "class_weight",
    "under_sample",
    # "smote",  # Memory-intensive — enable with caution
]

MODELS: List[str] = [
    "logistic_regression",
    "random_forest",
    # "lightgbm",  # Requires: pip install lightgbm
]

# Random-search grid for Phase 3.
HYPERPARAMS: Dict[str, Dict[str, List[Any]]] = {
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    },
}

# Fixed logistic regression params used in Phase 1 (balancing comparison).
_PHASE1_MODEL = "logistic_regression"
_PHASE1_PARAMS: Dict[str, Any] = {"C": 1.0, "max_iter": 1000, "random_state": 42}

# Phase 3: how many random param combos to try.
_N_RANDOM_ITERS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    path: str,
    target_col: str = "is_laundering",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Load a Parquet file and return a stratified train / test split.

    Converts float64 → float32 to halve memory for wide feature matrices.
    """
    logger.info("Loading data from %s", path)
    df = pd.read_parquet(path)

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    del df
    gc.collect()

    # Memory optimisation: float64 → float32
    f64_cols = X.select_dtypes(include="float64").columns
    if len(f64_cols):
        X[f64_cols] = X[f64_cols].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    del X, y
    gc.collect()

    logger.info(
        "Split: train=%d  test=%d  positive_rate_train=%.4f",
        len(X_train), len(X_test), y_train.mean(),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _record(
    phase: int,
    balancing: str,
    model: str,
    params: dict,
    result: Dict[str, Any],
    elapsed: float,
    n_train: int,
) -> Dict[str, Any]:
    """Build a flat results dict ready for a DataFrame row."""
    metrics = result["metrics"]
    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "balancing_method": balancing,
        "model_name": model,
        "params": json.dumps(params),
        "pr_auc": metrics.get("pr_auc", float("nan")),
        "roc_auc": metrics.get("roc_auc", float("nan")),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "wall_time_seconds": round(elapsed, 2),
        "n_train_samples": n_train,
        "status": "success",
    }
    # Carry through any recall_at_X%_precision columns.
    for k, v in metrics.items():
        if k.startswith("recall_at_"):
            row[k] = v
    return row


def _error_record(
    phase: int, balancing: str, model: str, params: dict, exc: Exception
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "balancing_method": balancing,
        "model_name": model,
        "params": json.dumps(params),
        "status": f"error: {exc}",
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_one(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    balancing: str,
    model_name: str,
    model_params: dict,
    random_state: int,
) -> tuple:
    """
    Apply balancing then train + evaluate.
    Returns (result_dict, elapsed_seconds, n_train_samples).
    Raises on any internal error (caller handles).
    """
    X_bal, y_bal, weights = balance_data(
        X_train, y_train, method=balancing, random_state=random_state
    )

    # class_weight passes weights through to the estimator
    effective_weights = weights if balancing == "class_weight" else None

    t0 = time.perf_counter()
    result = train_and_evaluate(
        X_bal, y_bal, X_test, y_test,
        model_name=model_name,
        model_params=model_params,
        sample_weights=effective_weights,
    )
    elapsed = time.perf_counter() - t0
    n_train = len(X_bal)

    del X_bal, y_bal
    gc.collect()

    return result, elapsed, n_train


def phase1_balancing(
    X_train, y_train, X_test, y_test,
    results: List[Dict],
    random_state: int = 42,
) -> str:
    """
    Compare balancing strategies with a fixed Logistic Regression model.
    Returns the name of the best-performing balancing method (by PR-AUC).
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 — Balancing Strategy Comparison")
    logger.info("=" * 60)

    for method in BALANCING_METHODS:
        logger.info("  Balancing: %s", method)
        try:
            result, elapsed, n_train = _run_one(
                X_train, y_train, X_test, y_test,
                method, _PHASE1_MODEL, _PHASE1_PARAMS, random_state,
            )
            row = _record(1, method, _PHASE1_MODEL, _PHASE1_PARAMS, result, elapsed, n_train)
            logger.info("  → PR-AUC=%.4f  Recall=%.4f  %.1fs", row["pr_auc"], row["recall"], elapsed)
        except Exception as exc:
            logger.error("  FAILED: %s", exc, exc_info=True)
            row = _error_record(1, method, _PHASE1_MODEL, _PHASE1_PARAMS, exc)
        results.append(row)

    best = _best_balancing(results)
    logger.info("Phase 1 best balancing: %s", best)
    return best


def phase2_models(
    X_train, y_train, X_test, y_test,
    best_balancing: str,
    results: List[Dict],
    random_state: int = 42,
) -> str:
    """
    Compare model families with the best balancing strategy from Phase 1.
    Returns the name of the best-performing model (by PR-AUC).
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 — Model Comparison  (balancing=%s)", best_balancing)
    logger.info("=" * 60)

    best_model: Optional[str] = None
    best_pr_auc = -1.0

    for model_name in MODELS:
        logger.info("  Model: %s", model_name)
        try:
            result, elapsed, n_train = _run_one(
                X_train, y_train, X_test, y_test,
                best_balancing, model_name, {}, random_state,
            )
            row = _record(2, best_balancing, model_name, {}, result, elapsed, n_train)
            logger.info("  → PR-AUC=%.4f  %.1fs", row["pr_auc"], elapsed)
            if row["pr_auc"] > best_pr_auc:
                best_pr_auc = row["pr_auc"]
                best_model = model_name
        except Exception as exc:
            logger.error("  FAILED: %s", exc, exc_info=True)
            row = _error_record(2, best_balancing, model_name, {}, exc)
        results.append(row)

    if best_model is None:
        raise RuntimeError("All models failed in Phase 2 — cannot continue.")

    logger.info("Phase 2 best model: %s  (PR-AUC=%.4f)", best_model, best_pr_auc)
    return best_model


def phase3_tuning(
    X_train, y_train, X_test, y_test,
    best_balancing: str,
    best_model: str,
    results: List[Dict],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Random hyperparameter search for the winning (model, balancing) pair.
    Returns the best param dict found.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3 — Hyperparameter Tuning  model=%s  balancing=%s", best_model, best_balancing)
    logger.info("=" * 60)

    if best_model not in HYPERPARAMS:
        logger.info("No param grid defined for %s — skipping Phase 3.", best_model)
        return {}

    grid = HYPERPARAMS[best_model]
    rng = np.random.default_rng(random_state)

    # Build random combos, always inject random_state for reproducibility.
    combos: List[Dict[str, Any]] = []
    for _ in range(_N_RANDOM_ITERS):
        combo: Dict[str, Any] = {k: rng.choice(v).item() for k, v in grid.items()}
        combo["random_state"] = random_state
        combos.append(combo)

    best_params: Dict[str, Any] = {}
    best_pr_auc = -1.0

    for i, params in enumerate(combos, 1):
        logger.info("  Iter %d/%d  params=%s", i, len(combos), params)
        try:
            result, elapsed, n_train = _run_one(
                X_train, y_train, X_test, y_test,
                best_balancing, best_model, params, random_state,
            )
            row = _record(3, best_balancing, best_model, params, result, elapsed, n_train)
            logger.info("  → PR-AUC=%.4f  %.1fs", row["pr_auc"], elapsed)
            if row["pr_auc"] > best_pr_auc:
                best_pr_auc = row["pr_auc"]
                best_params = params
        except Exception as exc:
            logger.error("  FAILED: %s", exc, exc_info=True)
            row = _error_record(3, best_balancing, best_model, params, exc)
        results.append(row)

    logger.info("Phase 3 best params: %s  (PR-AUC=%.4f)", best_params, best_pr_auc)
    return best_params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _best_balancing(results: List[Dict]) -> str:
    """Return the balancing method with the highest PR-AUC from Phase 1."""
    p1 = [r for r in results if r.get("phase") == 1 and r.get("status") == "success"]
    if not p1:
        logger.warning("No Phase 1 results — defaulting to 'none'.")
        return "none"
    return max(p1, key=lambda r: r.get("pr_auc", -1.0))["balancing_method"]


def _print_summary(results: List[Dict], best_balancing: str, best_model: str, best_params: dict) -> None:
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("  Best balancing : %s", best_balancing)
    logger.info("  Best model     : %s", best_model)
    logger.info("  Best params    : %s", best_params)
    successful = [r for r in results if r.get("status") == "success"]
    top5 = sorted(successful, key=lambda r: r.get("pr_auc", -1.0), reverse=True)[:5]
    logger.info("  Top-5 experiments by PR-AUC:")
    for rank, r in enumerate(top5, 1):
        logger.info(
            "    %d. %-15s + %-20s  PR-AUC=%.4f",
            rank, r["balancing_method"], r["model_name"], r.get("pr_auc", float("nan")),
        )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AML Experimentation Orchestrator")
    parser.add_argument("--data_path", default="aml_features/train_features.parquet")
    parser.add_argument("--output_path", default="outputs/experiment_results.csv")
    parser.add_argument("--target_col", default="Is Laundering")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args(argv)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load once, share across phases.
    X_train, X_test, y_train, y_test = load_data(
        args.data_path, args.target_col, args.test_size, args.random_state
    )

    results: List[Dict] = []

    best_balancing = phase1_balancing(X_train, y_train, X_test, y_test, results, args.random_state)
    best_model = phase2_models(X_train, y_train, X_test, y_test, best_balancing, results, args.random_state)
    best_params = phase3_tuning(X_train, y_train, X_test, y_test, best_balancing, best_model, results, args.random_state)

    pd.DataFrame(results).to_csv(args.output_path, index=False)
    logger.info("Results saved → %s", args.output_path)

    _print_summary(results, best_balancing, best_model, best_params)
    return 0


if __name__ == "__main__":
    sys.exit(main())