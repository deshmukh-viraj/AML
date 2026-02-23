# src/shap_explainability.py
"""
SHAP Explainability for AML — Memory-safe, 8 GB RAM compatible.

Responsibility:
    Produce three layers of explanation for a fitted tree model:

    1. Global importance — which features drive risk decisions overall?
    2. Local explanation  — why was *this* specific transaction flagged?
    3. Feature drift     — are SHAP distributions shifting over time?

    Counterfactuals (alibi-explain) are available as an optional extension
    but are NOT part of the default pipeline to keep dependencies minimal.

Design decisions:
    - Reservoir sampling limits the global analysis to ≤ 50 000 rows,
      keeping peak RAM under ~2 GB even on 30 M-row datasets.
    - Local SHAP is computed only for the top-N high-risk cases and a
      small borderline window — not for every transaction.
    - All heavy numpy arrays are deleted and gc.collect() is called after
      each stage so memory is freed promptly.
    - Drift detection uses PSI (Population Stability Index) on SHAP values,
      which is interpretable and threshold-free.

Supported model types: XGBoost, LightGBM, CatBoost, scikit-learn forests.
Any model compatible with ``shap.TreeExplainer`` will work.

Dependencies:
    shap>=0.44, numpy, pandas, matplotlib
    (optional) alibi for counterfactuals
"""

from __future__ import annotations

import gc
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Global analysis
    "global_sample_frac": 0.001,     # fraction of rows to use for global SHAP
    "global_max_samples": 50_000,    # hard cap regardless of fraction
    "random_state": 42,

    # Local analysis
    "top_n_risk": 100,               # highest-probability cases to explain
    "borderline_n": 50,              # cases within `borderline_margin` of threshold
    "decision_threshold": 0.5,
    "borderline_margin": 0.05,

    # Drift detection
    "drift_max_samples": 10_000,     # sub-sample for new-period SHAP in drift check
    "drift_psi_threshold": 0.10,     # PSI > 0.10 → potential drift

    # Output
    "output_dir": "outputs/shap",
    "save_plots": True,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _model_type(model: Any) -> Optional[str]:
    """Return a short type tag for logging; None means unsupported."""
    name = type(model).__name__.lower()
    for tag in ("xgb", "lgbm", "lightgbm", "catboost", "forest"):
        if tag in name:
            return tag
    return None


def reservoir_sample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_frac: float,
    max_samples: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Stratified reservoir sampling without loading the full dataset into memory.

    Streams through rows in a single pass, maintaining equal-class reservoirs.
    Result is capped at `max_samples` rows total.
    """
    rng = np.random.default_rng(random_state)
    n_per_class = min(int(len(X) * sample_frac), max_samples) // 2

    res_pos: List[int] = []
    res_neg: List[int] = []
    cnt_pos = cnt_neg = 0

    for i, label in enumerate(y.values):
        if label == 1:
            cnt_pos += 1
            if len(res_pos) < n_per_class:
                res_pos.append(i)
            elif rng.random() < n_per_class / cnt_pos:
                res_pos[int(rng.integers(n_per_class))] = i
        else:
            cnt_neg += 1
            if len(res_neg) < n_per_class:
                res_neg.append(i)
            elif rng.random() < n_per_class / cnt_neg:
                res_neg[int(rng.integers(n_per_class))] = i

    selected = sorted(set(res_pos + res_neg))
    logger.info("reservoir_sample: kept %d rows (%.2f%%)", len(selected), 100 * len(selected) / len(X))
    return X.iloc[selected].copy(), y.iloc[selected].copy()


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.
    PSI < 0.10  → stable
    PSI 0.10–0.25 → moderate drift
    PSI > 0.25  → significant drift
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    eps = 1e-6  # avoid log(0)
    exp_pct = np.histogram(expected, breakpoints)[0] / len(expected) + eps
    act_pct = np.histogram(actual, breakpoints)[0] / len(actual) + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _select_local_cases(
    X: pd.DataFrame,
    probs: np.ndarray,
    top_n: int,
    borderline_n: int,
    threshold: float,
    margin: float,
) -> pd.Index:
    """Union of highest-probability cases and borderline cases."""
    high_risk_pos = np.argsort(probs)[-top_n:]
    dist = np.abs(probs - threshold)
    borderline_pos = np.where(dist <= margin)[0]
    if len(borderline_pos) > borderline_n:
        borderline_pos = borderline_pos[np.argsort(dist[borderline_pos])[:borderline_n]]
    combined = np.union1d(high_risk_pos, borderline_pos)
    return X.index[combined]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AMLExplainer:
    """
    SHAP-based explainability for AML tree models.

    Instantiate once per model, then call the analysis methods.

    Parameters
    ----------
    model : fitted tree-based classifier
        Must be compatible with ``shap.TreeExplainer``.
    config : dict, optional
        Override any keys from ``DEFAULT_CONFIG``.
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None):
        if _model_type(model) is None:
            raise ValueError(
                f"{type(model).__name__} is not a recognised tree model. "
                "Use XGBoost, LightGBM, CatBoost, or a scikit-learn forest."
            )
        self.model = model
        self.cfg: Dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
        self._explainer = None          # lazy-initialised on first use
        self._ref_shap: Optional[np.ndarray] = None  # stored for drift baseline

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_explainer(self) -> None:
        if self._explainer is not None:
            return
        try:
            import shap  # type: ignore
        except ImportError as exc:
            raise ImportError("Install shap: pip install shap") from exc
        self._explainer = shap.TreeExplainer(self.model)
        logger.info("TreeExplainer initialised for %s", type(self.model).__name__)

    def _shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values for the positive class, shape (n, p)."""
        self._init_explainer()
        sv = self._explainer.shap_values(X)
        # Binary classifiers may return a list [neg_class, pos_class].
        if isinstance(sv, list):
            return sv[1]
        return sv

    def _output_dir(self) -> Path:
        p = Path(self.cfg["output_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Global analysis
    # ------------------------------------------------------------------

    def global_explain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute global feature importance via reservoir-sampled SHAP values.

        Saves ``global_importance.csv`` and (optionally) ``global_summary.png``
        to ``output_dir``.

        Returns
        -------
        pd.DataFrame
            Columns: feature, mean_abs_shap, std_shap, rank.
            Sorted descending by importance.
        """
        out = self._output_dir()

        X_s, y_s = reservoir_sample(
            X, y,
            sample_frac=self.cfg["global_sample_frac"],
            max_samples=self.cfg["global_max_samples"],
            random_state=self.cfg["random_state"],
        )
        logger.info("Computing global SHAP on %d samples ...", len(X_s))

        shap_vals = self._shap_values(X_s)
        self._ref_shap = shap_vals  # keep for drift baseline

        importance = (
            pd.DataFrame({
                "feature": X_s.columns,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
                "std_shap": shap_vals.std(axis=0),
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance["rank"] = importance.index + 1

        importance.to_csv(out / "global_importance.csv", index=False)
        logger.info("Global importance → %s", out / "global_importance.csv")

        if self.cfg["save_plots"]:
            self._save_summary_plot(shap_vals, X_s, out / "global_summary.png")

        del shap_vals
        gc.collect()

        return importance

    def _save_summary_plot(
        self, shap_vals: np.ndarray, X: pd.DataFrame, path: Path
    ) -> None:
        try:
            import shap  # type: ignore
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            logger.warning("matplotlib not installed — skipping summary plot.")
            return
        shap.summary_plot(shap_vals, X, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Summary plot → %s", path)

    # ------------------------------------------------------------------
    # Local analysis
    # ------------------------------------------------------------------

    def local_explain(
        self,
        X: pd.DataFrame,
        probs: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute per-case SHAP values for high-risk and borderline transactions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for the inference set.
        probs : np.ndarray, optional
            Model probabilities for the positive class.
            Computed from ``self.model`` if not provided.

        Returns
        -------
        pd.DataFrame
            One row per explained case: probability, shap_sum, case_type,
            plus one column per feature with its SHAP contribution.
        """
        out = self._output_dir()

        if probs is None:
            probs = self.model.predict_proba(X)[:, 1]

        target_idx = _select_local_cases(
            X, probs,
            top_n=self.cfg["top_n_risk"],
            borderline_n=self.cfg["borderline_n"],
            threshold=self.cfg["decision_threshold"],
            margin=self.cfg["borderline_margin"],
        )
        logger.info("Local SHAP: explaining %d cases", len(target_idx))

        X_target = X.loc[target_idx]
        probs_target = probs[target_idx.get_loc(target_idx)]  # aligned subset

        # Safer index alignment when using iloc-based operations.
        positional = [X.index.get_loc(i) for i in target_idx]
        probs_target = probs[positional]

        shap_vals = self._shap_values(X_target)

        expected_val = self._explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = expected_val[1]

        hi_threshold = np.sort(probs)[-self.cfg["top_n_risk"]]

        results = pd.DataFrame(
            {
                "probability": probs_target,
                "shap_sum": shap_vals.sum(axis=1),
                "shap_base": float(expected_val),
                "case_type": [
                    "high_risk"
                    if p >= hi_threshold
                    else (
                        "borderline"
                        if abs(p - self.cfg["decision_threshold"]) <= self.cfg["borderline_margin"]
                        else "normal"
                    )
                    for p in probs_target
                ],
            },
            index=target_idx,
        )

        shap_df = pd.DataFrame(shap_vals, index=target_idx, columns=X.columns)
        full = pd.concat([results, shap_df], axis=1)
        full.to_csv(out / "local_explanations.csv")
        logger.info("Local explanations → %s", out / "local_explanations.csv")

        del shap_vals, shap_df
        gc.collect()

        return full

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        X_new: pd.DataFrame,
        X_reference: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Detect distributional drift in SHAP values using PSI.

        Parameters
        ----------
        X_new : pd.DataFrame
            Recent transactions to check for drift.
        X_reference : pd.DataFrame, optional
            Historical reference data.  If ``None``, the SHAP array stored
            by the last ``global_explain`` call is used.

        Returns
        -------
        dict
            Maps feature name → PSI score.
            Also logs a warning for any feature exceeding the PSI threshold.
        """
        out = self._output_dir()

        if X_reference is None and self._ref_shap is None:
            raise RuntimeError(
                "No reference SHAP values available. "
                "Call global_explain() first, or pass X_reference."
            )

        # Reference SHAP
        if X_reference is not None:
            ref_shap = self._shap_values(X_reference)
        else:
            ref_shap = self._ref_shap  # type: ignore[assignment]

        # New SHAP — sub-sample to keep RAM bounded
        if len(X_new) > self.cfg["drift_max_samples"]:
            X_new = X_new.sample(
                self.cfg["drift_max_samples"], random_state=self.cfg["random_state"]
            )
        new_shap = self._shap_values(X_new)

        psi_scores: Dict[str, float] = {}
        for i, feature in enumerate(X_new.columns):
            psi_scores[feature] = _psi(ref_shap[:, i], new_shap[:, i])

        drifted = [f for f, s in psi_scores.items() if s > self.cfg["drift_psi_threshold"]]
        if drifted:
            warnings.warn(
                f"SHAP drift detected in {len(drifted)} features: {drifted[:10]}",
                UserWarning,
            )

        pd.Series(psi_scores).to_json(out / "drift_psi_scores.json")
        summary = {
            "drifted_features": drifted,
            "max_psi": max(psi_scores.values()),
            "mean_psi": float(np.mean(list(psi_scores.values()))),
        }
        (out / "drift_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(
            "Drift check: %d features, max PSI=%.4f, %d above threshold",
            len(psi_scores), summary["max_psi"], len(drifted),
        )

        del new_shap
        gc.collect()

        return psi_scores

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_inference: pd.DataFrame,
        probs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run global → local → drift in sequence.

        Parameters
        ----------
        X_train, y_train : training data (used for global importance + drift ref)
        X_inference : data for which you want local explanations
        probs : pre-computed positive-class probabilities for X_inference

        Returns
        -------
        dict with keys: ``global``, ``local``, ``drift``.
        """
        logger.info("=== AML SHAP Full Analysis ===")

        global_result = self.global_explain(X_train, y_train)
        local_result = self.local_explain(X_inference, probs)
        drift_result = self.detect_drift(X_inference)  # uses stored ref from global_explain

        logger.info("=== Analysis Complete ===")

        return {
            "global": global_result,
            "local": local_result,
            "drift": drift_result,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def explain_aml_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_inference: pd.DataFrame,
    probs: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    One-call entry point for end-to-end SHAP analysis.

    Example
    -------
    >>> results = explain_aml_model(
    ...     model=xgb_clf,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_inference=X_daily,
    ...     config={"output_dir": "outputs/2025-01-15"},
    ... )
    """
    explainer = AMLExplainer(model, config)
    return explainer.run_full_analysis(X_train, y_train, X_inference, probs)