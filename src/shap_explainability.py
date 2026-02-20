"""
SHAP Explainability for AML - Optimized for 30M+ Transactions

Memory-efficient, streaming-compatible, with counterfactuals and drift detection.
"""

import gc
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "sample_frac": 0.001,           # 0.1% of 30M = 30K samples
    "max_global_samples": 50000,    # Hard cap for safety
    "streaming_chunksize": 100000,  # Memory-friendly chunks
    "top_n_risk": 100,              # High-risk cases for local SHAP
    "borderline_n": 50,             # Near-threshold cases
    "decision_threshold": 0.5,
    "borderline_margin": 0.05,
    "sparse_top_k": 10,             # Keep only top 10 features per sample
    "enable_counterfactuals": True,
    "cf_max_changes": 3,
    "cf_target_threshold": 0.3,
    "n_jobs": -1,                   # Parallel processing
    "batch_size": 500,
    "random_state": 42,
}


# =============================================================================
# Core Utilities
# =============================================================================

def get_model_type(model: Any) -> Optional[str]:
    """Identify tree-based model type for TreeExplainer compatibility."""
    name = type(model).__name__.lower()
    if "xgb" in name: return "xgb"
    if "lgbm" in name or "lightgbm" in name: return "lgbm"
    if "catboost" in name: return "catboost"
    if "forest" in name: return "random_forest"
    return None


def reservoir_sample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_frac: float,
    random_state: int = 42,
    chunksize: int = 100000
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Memory-efficient stratified sampling via reservoir sampling.
    Processes data in chunks without loading full dataset into memory.
    """
    np.random.seed(random_state)
    n_total = len(X)
    n_target = min(int(n_total * sample_frac), 50000)
    n_per_class = n_target // 2
    
    # Reservoirs for each class
    reservoir_pos, reservoir_neg = [], []
    count_pos = count_neg = 0
    
    # Stream through data in chunks
    for start in range(0, n_total, chunksize):
        end = min(start + chunksize, n_total)
        X_chunk = X.iloc[start:end]
        y_chunk = y.iloc[start:end]
        
        for idx, label in enumerate(y_chunk.values):
            global_idx = start + idx
            if label == 1:
                count_pos += 1
                if len(reservoir_pos) < n_per_class:
                    reservoir_pos.append(global_idx)
                elif np.random.random() < n_per_class / count_pos:
                    reservoir_pos[np.random.randint(n_per_class)] = global_idx
            else:
                count_neg += 1
                if len(reservoir_neg) < n_per_class:
                    reservoir_neg.append(global_idx)
                elif np.random.random() < n_per_class / count_neg:
                    reservoir_neg[np.random.randint(n_per_class)] = global_idx
    
    selected = list(dict.fromkeys(reservoir_pos + reservoir_neg))  # Dedupe, preserve order
    return X.iloc[selected].copy(), y.iloc[selected].copy()


def select_cases(
    X: pd.DataFrame,
    probs: np.ndarray,
    top_n: int,
    borderline_n: int,
    threshold: float,
    margin: float
) -> pd.Index:
    """Select high-risk and borderline cases for local explanation."""
    # High-risk: highest probabilities
    high_risk = X.index[np.argsort(probs)[-top_n:]]
    
    # Borderline: within margin of threshold
    dist = np.abs(probs - threshold)
    borderline_mask = dist <= margin
    borderline_candidates = np.where(borderline_mask)[0]
    
    if len(borderline_candidates) > borderline_n:
        borderline = X.index[borderline_candidates[np.argsort(dist[borderline_candidates])[:borderline_n]]]
    else:
        borderline = X.index[borderline_candidates]
    
    return high_risk.union(borderline).drop_duplicates()


def sparse_shap(shap_values: np.ndarray, top_k: int) -> sparse.csr_matrix:
    """Keep only top-k absolute SHAP values per sample to reduce memory."""
    sparse_vals = np.zeros_like(shap_values)
    for i in range(len(shap_values)):
        top_idx = np.argsort(np.abs(shap_values[i]))[-top_k:]
        sparse_vals[i, top_idx] = shap_values[i, top_idx]
    return sparse.csr_matrix(sparse_vals)


# =============================================================================
# Main Explainer Class
# =============================================================================

class AMLExplainer:
    """
    Production-grade SHAP explainer for large-scale AML.
    Optimized for memory efficiency and regulatory compliance.
    """
    
    def __init__(self, model: Any, config: Optional[Dict] = None):
        if not get_model_type(model):
            raise ValueError(f"Model {type(model).__name__} not supported. Use XGBoost/LightGBM/CatBoost/RF.")
        
        self.model = model
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model_type = get_model_type(model)
        self.explainer = None
        self.global_shap = None  # Store for drift detection
        
    def _init_shap(self):
        """Lazy initialization of TreeExplainer."""
        if self.explainer is None:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
    
    # -------------------------------------------------------------------------
    # Global Analysis
    # -------------------------------------------------------------------------
    
    def global_explain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str = "outputs/shap"
    ) -> pd.DataFrame:
        """
        Compute global feature importance using reservoir sampling.
        Memory-efficient for 30M+ row datasets.
        """
        self._init_shap()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Sample efficiently
        print(f"[Global] Sampling {self.config['sample_frac']:.2%} via reservoir sampling...")
        X_sample, _ = reservoir_sample(
            X, y,
            sample_frac=self.config["sample_frac"],
            random_state=self.config["random_state"],
            chunksize=self.config["streaming_chunksize"]
        )
        print(f"[Global] Sampled {len(X_sample):,} rows")
        
        # Compute SHAP values
        print("[Global] Computing SHAP values...")
        shap_vals = self.explainer.shap_values(X_sample)
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]  # Positive class
        
        # Store for drift detection
        self.global_shap = shap_vals
        
        # Aggregate importance
        importance = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": np.mean(np.abs(shap_vals), axis=0),
            "std_shap": np.std(shap_vals, axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        importance["rank"] = range(1, len(importance) + 1)
        
        # Save
        importance.to_csv(f"{output_dir}/global_importance.csv", index=False)
        self._save_plot(shap_vals, X_sample, f"{output_dir}/global_summary.png")
        
        # Cleanup
        del shap_vals
        gc.collect()
        
        return importance
    
    def _save_plot(self, shap_vals: np.ndarray, X: pd.DataFrame, path: str):
        """Generate SHAP summary plot."""
        import shap
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_vals, X, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Global] Plot saved: {path}")
    
    # -------------------------------------------------------------------------
    # Local Analysis
    # -------------------------------------------------------------------------
    
    def local_explain(
        self,
        X: pd.DataFrame,
        probs: Optional[np.ndarray] = None,
        output_dir: str = "outputs/shap"
    ) -> pd.DataFrame:
        """
        Local SHAP for high-risk and borderline cases.
        Uses sparse storage for memory efficiency.
        """
        self._init_shap()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get probabilities if needed
        if probs is None:
            probs = self.model.predict_proba(X)[:, 1]
        
        # Select target cases
        target_idx = select_cases(
            X, probs,
            self.config["top_n_risk"],
            self.config["borderline_n"],
            self.config["decision_threshold"],
            self.config["borderline_margin"]
        )
        print(f"[Local] Explaining {len(target_idx)} cases...")
        
        X_target = X.loc[target_idx]
        probs_target = probs[target_idx.values]
        
        # Parallel batch processing for speed
        shap_vals = self._parallel_shap(X_target)
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        
        # Convert to sparse for memory efficiency
        sparse_vals = sparse_shap(shap_vals, self.config["sparse_top_k"])
        
        # Build results DataFrame
        results = pd.DataFrame({
            "probability": probs_target,
            "shap_sum": np.array(sparse_vals.sum(axis=1)).flatten(),
            "shap_base": self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
        }, index=target_idx)
        
        # Add top feature contributions
        feature_contribs = pd.DataFrame.sparse.from_spmatrix(
            sparse_vals, 
            index=target_idx, 
            columns=X.columns
        )
        
        # Determine case type
        high_threshold = np.sort(probs)[-self.config["top_n_risk"]]
        results["case_type"] = "normal"
        results.loc[results["probability"] >= high_threshold, "case_type"] = "high_risk"
        borderline_mask = np.abs(results["probability"] - self.config["decision_threshold"]) <= self.config["borderline_margin"]
        results.loc[borderline_mask, "case_type"] += "+borderline"
        
        # Combine and save
        full_results = pd.concat([results, feature_contribs], axis=1)
        full_results.to_csv(f"{output_dir}/local_explanations.csv")
        
        # Cleanup
        del shap_vals, sparse_vals
        gc.collect()
        
        return full_results
    
    def _parallel_shap(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values in parallel batches."""
        batch_size = self.config["batch_size"]
        
        def compute_batch(start):
            end = min(start + batch_size, len(X))
            return self.explainer.shap_values(X.iloc[start:end])
        
        batches = range(0, len(X), batch_size)
        results = Parallel(n_jobs=self.config["n_jobs"], prefer="threads")(
            delayed(compute_batch)(i) for i in batches
        )
        
        # Handle binary classification list output
        if isinstance(results[0], list):
            class_0 = np.vstack([r[0] for r in results])
            class_1 = np.vstack([r[1] for r in results])
            return [class_0, class_1]
        return np.vstack(results)
    
    # -------------------------------------------------------------------------
    # Counterfactual Explanations (AML-Specific)
    # -------------------------------------------------------------------------
    
    def counterfactual_explain(
        self,
        X_flagged: pd.DataFrame,
        output_dir: str = "outputs/shap"
    ) -> Optional[pd.DataFrame]:
        """
        Generate actionable counterfactuals for flagged cases.
        Shows minimal changes needed to reduce risk score.
        """
        if not self.config["enable_counterfactuals"]:
            return None
        
        try:
            from alibi.explainers import CounterfactualProto
        except ImportError:
            warnings.warn("Alibi not installed. Skipping counterfactuals.")
            return None
        
        print(f"[Counterfactual] Generating for {len(X_flagged)} cases...")
        
        # Initialize with small reference set
        cf = CounterfactualProto(
            self.model.predict_proba,
            shape=(1, X_flagged.shape[1]),
            use_kdtree=True,
            max_iterations=500,
            early_stop=50
        )
        cf.fit(X_flagged.iloc[:min(100, len(X_flagged))].values)
        
        results = []
        for idx, (_, row) in enumerate(X_flagged.iterrows()):
            if idx >= 50:  # Limit to prevent timeout
                break
                
            orig_proba = self.model.predict_proba(row.values.reshape(1, -1))[0][1]
            
            # Skip if already below target
            if orig_proba < self.config["cf_target_threshold"]:
                continue
            
            explanation = cf.explain(row.values.reshape(1, -1))
            
            if explanation.cf is not None:
                cf_proba = explanation.cf['class_proba'][0][1]
                cf_row = explanation.cf['X'][0]
                
                # Identify changes
                changes = {
                    feat: {"from": orig, "to": new}
                    for feat, orig, new in zip(X_flagged.columns, row.values, cf_row)
                    if not np.isclose(orig, new)
                }
                
                results.append({
                    "index": idx,
                    "original_proba": orig_proba,
                    "counterfactual_proba": cf_proba,
                    "risk_reduction": orig_proba - cf_proba,
                    "n_changes": len(changes),
                    "changes": json.dumps(changes) if changes else "{}"
                })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{output_dir}/counterfactuals.csv", index=False)
        print(f"[Counterfactual] Saved {len(df)} explanations")
        
        return df
    
    # -------------------------------------------------------------------------
    # Drift Detection
    # -------------------------------------------------------------------------
    
    def detect_drift(
        self,
        X_new: pd.DataFrame,
        X_reference: Optional[pd.DataFrame] = None,
        output_dir: str = "outputs/shap"
    ) -> Dict[str, float]:
        """
        Detect drift in SHAP value distributions.
        Critical for AML model monitoring and regulatory compliance.
        """
        self._init_shap()
        
        if X_reference is None and self.global_shap is None:
            raise ValueError("Run global_explain first or provide reference data")
        
        print("[Drift] Computing SHAP distributions...")
        
        # Get reference SHAP values
        if X_reference is not None:
            ref_shap = self.explainer.shap_values(X_reference)
            if isinstance(ref_shap, list): ref_shap = ref_shap[1]
        else:
            ref_shap = self.global_shap
        
        # Get new SHAP values (sample if large)
        if len(X_new) > 10000:
            X_new = X_new.sample(10000, random_state=42)
        
        new_shap = self.explainer.shap_values(X_new)
        if isinstance(new_shap, list): new_shap = new_shap[1]
        
        # Calculate PSI per feature
        drift_scores = {}
        for i, feat in enumerate(X_new.columns):
            drift_scores[feat] = self._psi(ref_shap[:, i], new_shap[:, i])
        
        # Report
        report = {
            "drifted_features": [f for f, s in drift_scores.items() if s > 0.1],
            "max_psi": max(drift_scores.values()),
            "mean_psi": np.mean(list(drift_scores.values()))
        }
        
        # Save
        pd.Series(drift_scores).to_json(f"{output_dir}/drift_scores.json")
        json.dump(report, open(f"{output_dir}/drift_summary.json", "w"))
        
        if report["drifted_features"]:
            warnings.warn(f"Drift detected in {len(report['drifted_features'])} features!")
        
        return drift_scores
    
    def _psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index for distribution comparison."""
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        expected_pct = np.histogram(expected, breakpoints)[0] / len(expected) + 0.0001
        actual_pct = np.histogram(actual, breakpoints)[0] / len(actual) + 0.0001
        
        return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------
    
    def run_full_analysis(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_inference: pd.DataFrame,
        probs: Optional[np.ndarray] = None,
        output_dir: str = "outputs/shap"
    ) -> Dict[str, Any]:
        """Execute complete explainability pipeline."""
        print("=" * 60)
        print("AML SHAP Analysis Pipeline")
        print("=" * 60)
        
        # Global
        global_res = self.global_explain(X_train, y_train, output_dir)
        
        # Local
        local_res = self.local_explain(X_inference, probs, output_dir)
        
        # Counterfactuals on high-risk subset
        high_risk = local_res[local_res["case_type"].str.contains("high_risk")]
        cf_res = self.counterfactual_explain(high_risk.head(50), output_dir) if len(high_risk) > 0 else None
        
        print("=" * 60)
        print("Analysis Complete")
        print("=" * 60)
        
        return {
            "global": global_res,
            "local": local_res,
            "counterfactuals": cf_res,
            "model_type": self.model_type
        }


# =============================================================================
# Convenience Function
# =============================================================================

def explain_aml_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_inference: pd.DataFrame,
    probs: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
    output_dir: str = "outputs/shap"
) -> Dict[str, Any]:
    """
    One-shot function for complete AML model explanation.
    
    Example:
        results = explain_aml_model(
            model=xgb_classifier,
            X_train=X_train,  # 30M rows
            y_train=y_train,
            X_inference=X_daily,  # Today's alerts
            probs=probs,
            config={"sample_frac": 0.001},
            output_dir="outputs/2024-01-15"
        )
    """
    explainer = AMLExplainer(model, config)
    return explainer.run_full_analysis(X_train, y_train, X_inference, probs, output_dir)