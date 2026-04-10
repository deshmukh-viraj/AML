import gc
import os
from sys import exception
import yaml
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as LR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

try: 
    from src.logger import logging
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

ZERO_IMPUTE_FEATURES = [
    "burst_score_1h", "burst_count_24h", "txn_in_hour",
    "toxic_corridor_count_28d", "toxic_corridor_volume_28d",
    "txn_count_28d", "txn_count_total",
    "total_amount_paid_28d", "total_amount_received_28d",
    "amount_paid_last_100",
    "flag_high_burst", "flag_large_gap",
    "flag_extreme_consistency", "flag_high_concentration",
    "flag_heavy_structuring", "anomaly_cascade_score",
    "cascade_frequency_28d",
]

#load params
def load_params(params_path: str='params.yaml')-> dict:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Parameters loaded from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"params.yaml file not found at {params_path}")
        raise


#features helper
def get_features(schema: dict, target_col: str, time_col: str) -> List[str]:
    """
    select numeric feaaatures from parquet schema
    excludes target and timestamp cols
    """
    num_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
        pl.UInt8, pl.UInt32, pl.UInt16, pl.UInt64,
        pl.Float32, pl.Float64, pl.Boolean
    )

    features = [
        col for col, dtype in schema.items() if isinstance(dtype, num_types) and col != target_col
    ]
    logger.info(f"Found {len(features)} numeric features")
    if "anomaly_score" not in features:
        logger.warning("anomaly_score not found")
    return features


def load_split(
    path: Path,
    features: List[str],
    target_col: str,
    time_col: str, 
    max_rows: int = None, 
    random_state: int = 42
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    load feature parquet file into numpy array
    """
    cols_to_load = list(set(features + [target_col, time_col]))
    n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()

    if max_rows and n_rows > max_rows:
        lazy = pl.scan_parquet(path).select(cols_to_load)
        fraud_df = lazy.filter(pl.col(target_col) == 1).collect()
        n_legit = min(max_rows - len(fraud_df), n_rows - len(fraud_df))
        rng = np.random.RandomState(random_state)
        n_legit_total = n_rows - len(fraud_df)
        legit_idx = sorted(rng.choice(n_legit_total, n_legit, replace=False).tolist())
        legit_df = (
            lazy.filter(pl.col(target_col) == 0)
            .with_row_index("_idx")
            .filter(pl.col("_idx").is_in(legit_idx))
            .drop("_idx")
            .collect()
        )
        df = pl.concat([fraud_df, legit_df]).sample(
            fraction=1.0, shuffle=True, seed=random_state
        )
        logger.info(
            f"Loaded {path.stem}: {len(df):,} rows"
            f"(all {len(fraud_df):,} fraud + {n_legit:,} legit sampled)"
        )
        del fraud_df, legit_df
        gc.collect()
    else:
        logger.info(f"Loading {path.stem}: all {n_rows:,} rows")
        df = pl.scan_parquet(path).select(cols_to_load).collect()

    logger.info(f"  RAM: {df.estimated_size() / 1024**2:.1f} MB")

    timestamps = df[time_col].cast(pl.Int64).to_numpy() if time_col in df.columns else None
    y = df[target_col].to_numpy().astype(np.int32)
    X = df.select([pl.col(c).cast(pl.Float32) for c in features]).to_numpy()

    del df
    gc.collect()

    #impute nulls strategy depends on feature type
    for col_idx, col_name in enumerate(features):
        row_indices = np.where(np.isnan(X[:, col_idx]))[0]
        if len(row_indices) == 0:
            continue
        if col_name in ZERO_IMPUTE_FEATURES:
            X[row_indices, col_idx] = 0.0
        else:
            col_mean = float(np.nanmean(X[:, col_idx]))
            X[row_indices, col_idx] = col_mean if not np.isnan(col_mean) else 0.0

    n_fraud = int(y.sum())
    logger.info(f"fraud={n_fraud:,} | legit={len(y)-n_fraud:,}")

    return X, y, timestamps
    
# metrics

def recall_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    return float(np.interp(target, fpr, tpr))
 
 
def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, fpr_threshold: float) -> Dict:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmin(np.abs(fpr - fpr_threshold))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc":  float(auc(recall, precision)),
        "recall_at_1pct_fpr": recall_at_fpr(fpr, tpr, fpr_threshold),
        "fpr_at_threshold":float(fpr[idx]),
        "threshold_1pct_fpr": float(thresholds[idx]) if idx < len(thresholds) else 1.0,
    }
 
 
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, fpr_threshold: float) -> float:
    """
    find threshold that maximises recall subject to FPR <= business constraint.
    this is the operational threshold saved with the model.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_mask = fpr <= fpr_threshold
    if not valid_mask.any():
        logger.warning("No threshold achieves target FPR defaulting to 0.5")
        return 0.5
    best_idx = np.argmax(tpr[valid_mask])
    optimal_threshold = float(thresholds[valid_mask][best_idx])
    logger.info(
        f"Optimal threshold: {optimal_threshold:.4f} | "
        f"Recall: {tpr[valid_mask][best_idx]:.4f} | "
        f"FPR: {fpr[valid_mask][best_idx]:.4f}"
    )
    return optimal_threshold
 

# MCCV
def temporal_mccv_split(
    timestamps: np.ndarray,
    fold_i: int,
    n_folds: int = 5,
    val_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    expanding window walk-forward split.
    train grows from day 1, val window slides forward.
    """
    sorted_idx = np.argsort(timestamps)
    n = len(sorted_idx)
    val_size = int(val_frac * n)
    usable_train_end = n - (n_folds * val_size)
 
    if usable_train_end <= 0:
        raise ValueError(f"Not enough data for {n_folds} folds with val_frac={val_frac}")
 
    train_end = usable_train_end + fold_i * val_size
    val_start = train_end
    val_end = val_start + val_size
 
    return sorted_idx[:train_end], sorted_idx[val_start:val_end]
 
 
# model configs
def get_model_configs(imbalance_ratio: float, random_state: int) -> Dict:
    return {
        "XGBoost": {
            "class": XGBClassifier,
            "params": {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "scale_pos_weight": imbalance_ratio * 0.8,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "random_state": random_state,
                "n_jobs":-1,
            },
            "preprocessor": None,
        },
        "LightGBM": {
            "class": lgb.LGBMClassifier,
            "params": {
                "n_estimators":400,
                "max_depth":6,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "scale_pos_weight": imbalance_ratio * 0.6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state":random_state,
                "n_jobs": -1,
                "verbose":-1,
            },
            "preprocessor": None,
        },
        "RandomForest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators":200,
                "max_depth":10,
                "class_weight": "balanced",
                "min_samples_split": 50,
                "max_samples": 200_000,
                "random_state": random_state,
                "n_jobs": -1,
            },
            "preprocessor": None,
        },
        "LogisticRegression": {
            "class": LogisticRegression,
            "params": {
                "class_weight": "balanced",
                "max_iter": 1000,
                "C":0.1,
                "random_state": random_state,
                "n_jobs": -1,
            },
            "preprocessor": RobustScaler(),
        },
    }
 
 
def build_model(config: dict):
    """wrap model in scaling pipeline only if preprocessor specified."""
    if config.get("preprocessor") is not None:
        return Pipeline([
            ("scaler", clone(config["preprocessor"])),
            ("clf", config["class"](**config["params"]))
        ])
    return config["class"](**config["params"])
 
 
def mccv_evaluate(
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    n_folds: int,
    random_state: int,
    fpr_threshold: float,
    min_fraud_warn: int = 10
) -> Dict:
    """
    expanding window walk-forward MCCV.
    conservative score (mean - 2*std) used for model selection —
    """
    scores, thresholds = [], []
    logger.info(f"Running {n_folds} expanding walk-forward folds...")
 
    for i in range(n_folds):
        train_idx, val_idx = temporal_mccv_split(timestamps, fold_i=i, n_folds=n_folds)
 
        n_train_fraud = int(y[train_idx].sum())
        n_val_fraud = int(y[val_idx].sum())
 
        logger.info(
            f" Fold {i}: train={len(train_idx):,} ({n_train_fraud} fraud) |"
            f"val={len(val_idx):,} ({n_val_fraud} fraud)"
        )
 
        if n_train_fraud < min_fraud_warn or n_val_fraud < min_fraud_warn:
            logger.warning(f" Fold {i}: too few fraud cases — skipping")
            continue
 
        try:
            model = build_model(config)
            model.fit(X[train_idx], y[train_idx])
            probs= model.predict_proba(X[val_idx])[:, 1]
            metrics = calculate_metrics(y[val_idx], probs, fpr_threshold)
            scores.append(metrics["recall_at_1pct_fpr"])
            thresholds.append(metrics["threshold_1pct_fpr"])
            logger.info(f" Fold {i}: recall@1%FPR={metrics['recall_at_1pct_fpr']:.4f}")
            del model, probs
            gc.collect()
        except Exception as e:
            logger.warning(f" Fold {i} failed: {e}")
 
    if len(scores) < 3:
        raise ValueError(f"Only {len(scores)} valid folds — need at least 3")
 
    return {
        "mean_recall": float(np.mean(scores)),
        "std_recall": float(np.std(scores)),
        "conservative_score": float(np.mean(scores) - 2 * np.std(scores)),
        "mean_threshold":float(np.mean(thresholds)),
        "all_scores": scores,
        "n_valid_folds": len(scores),
    }
 
 
def select_best_model(mccv_results: Dict) -> Tuple[str, Dict]:
    """select model with highest conservative score (mean - 2*std)."""
    valid = {
        name: res for name, res in mccv_results.items()
        if res.get("n_valid_folds", 0) >= 3
    }
    if not valid:
        raise ValueError("No model produced 3+ valid MCCV folds")
 
    best_name= max(valid, key=lambda n: valid[n]["conservative_score"])
    best_result = valid[best_name]
 
    logger.info(
        f"Best model: {best_name} | "
        f"recall {best_result['mean_recall']:.4f} ± {best_result['std_recall']:.4f} | "
        f"conservative: {best_result['conservative_score']:.4f}"
    )
    return best_name, best_result
 
 
# calibration
def fit_platt_scaler(model, X_cal: np.ndarray, y_cal: np.ndarray) -> LR:
    """
    fit Platt scaling on calibration slice.
    corrects probability scores which are unreliable from tree models
    due to class imbalance and leaf node statistics.
    fit on cal slice only never on eval or test.
    """
    raw_scores = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    platt = LR()
    platt.fit(raw_scores, y_cal)
    logger.info("Platt scaling fitted on calibration slice")
    return platt
 
 
def calibrated_predict(model, platt: LR, X: np.ndarray) -> np.ndarray:
    raw_scores = model.predict_proba(X)[:, 1].reshape(-1, 1)
    return platt.predict_proba(raw_scores)[:, 1]
 
 
def check_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    shap_dir: Path
) -> float:
    """reliability diagram + calibration error metric."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
 
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, "o-", label=model_name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.title(f"Reliability Diagram  {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(shap_dir / f"{model_name}_calibration.png", dpi=150)
    plt.close()
 
    cal_error = float(np.mean(np.abs(prob_true - prob_pred)))
    logger.info(f"Calibration error: {cal_error:.4f} (< 0.05 good, > 0.10 concerning)")
    if cal_error > 0.10:
        logger.warning("Calibration error > 0.10  probability scores may be unreliable for analysts")
    return cal_error
 

# explainability
def explain_and_select_features(
    model,
    X_sample: np.ndarray,
    features: List[str],
    model_name: str,
    shap_dir: Path,
    shap_threshold: float = 0.001
) -> Tuple[np.ndarray, List[str]]:
    """
    generate SHAP plots and return SHAP-selected feature list.
    SHAPbased selection is used instead of RFE because:
    """
    logger.info(f"Generating SHAP plots for {model_name}...")
 
    try:
        act_model = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        X_input = model.named_steps["scaler"].transform(X_sample) if hasattr(model, "named_steps") else X_sample
 
        explainer = shap.TreeExplainer(act_model)
        shap_values = explainer.shap_values(X_input)
 
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
 
        shap.summary_plot(shap_values, X_input, feature_names=features, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(shap_dir / f"{model_name}_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
 
        shap.summary_plot(shap_values, X_input, feature_names=features, show=False, max_display=20, plot_type="bar")
        plt.tight_layout()
        plt.savefig(shap_dir / f"{model_name}_shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
 
        logger.info(f"SHAP plots saved to {shap_dir}")
 
        mean_shap = np.abs(shap_values).mean(axis=0)
        kept = [f for f, s in zip(features, mean_shap) if s >= shap_threshold]
        removed = [f for f, s in zip(features, mean_shap) if s < shap_threshold]
 
        if removed:
            logger.info(f"SHAP suggests removing {len(removed)} near-zero features: {removed}")
        logger.info(f"SHAP kept {len(kept)}/{len(features)} features — saved for next run")
 
        return mean_shap, kept
 
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return np.ones(len(features)), features
 
 
def plot_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    fpr_threshold: float,
    shap_dir: Path
) -> None:
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _= roc_curve(y_true, y_prob)
 
    fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
 
    ax1.plot(thresh, prec[:-1], label="Precision")
    ax1.plot(thresh, rec[:-1], label="Recall")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.set_title(f"{model_name}: Threshold vs Precision/Recall")
    ax1.grid(True, alpha=0.3)
 
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax2.axvline(x=fpr_threshold, color="r", linestyle="--", alpha=0.5, label=f"{fpr_threshold*100:.0f}% FPR target")
    ax2.fill_between([0, fpr_threshold], 0, 1, alpha=0.1, color="green")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    ax2.set_title("ROC Curve")
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(shap_dir / f"{model_name}_thresholds.png", dpi=150, bbox_inches="tight")
    plt.close()
 

# main
def main():
    logger.info("=" * 60)
    logger.info("Stage 4: Model Building")
    logger.info("=" * 60)
 
    try:
        params = load_params("params.yaml")
        model_cfg = params["model"]
        random_state = model_cfg.get("random_state", 42)
        fpr_threshold = model_cfg.get("fpr_threshold", 0.01)
        n_folds = model_cfg.get("mccv_iterations", 5)
        target_col= params["data_ingestion"].get("target_col", "Is Laundering")
        time_col= params["data_ingestion"].get("timestamp_col", "Timestamp")
 
        features_dir = Path(params["storage"].get("anomaly_dir", params['storage']['features_dir']))
        model_dir = Path(params["storage"]["model_build_dir"])
        shap_dir = model_dir / "shap"
        reports_dir= Path(params["storage"]["reports_dir"])
 
        model_dir.mkdir(parents=True, exist_ok=True)
        shap_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
 
        train_path = features_dir / "train_features.parquet"
        val_path = features_dir / "val_features.parquet"
        test_path = features_dir / "test_features.parquet"
 
        for p in [train_path, val_path, test_path]:
            if not p.exists():
                raise FileNotFoundError(f"{p} not found. Run feature_engineering stage first.")
 
        # feature list from schema
        schema = pl.scan_parquet(train_path).collect_schema()
        features = get_features(schema, target_col, time_col)
 
        #load data 
        logger.info("Loading training data...")
        X_train, y_train, train_ts = load_split(
            train_path, features, target_col, time_col,
            max_rows=model_cfg.get("train_sample_size", 1_500_000),
            random_state=random_state
        )
 
        logger.info("Loading validation data...")
        X_val, y_val, _ = load_split(
            val_path, features, target_col, time_col,
            max_rows=model_cfg.get("val_sample_size", 500_000),
            random_state=random_state
        )
 
        # variance filter  same mask applied to both train and val
        variances = np.var(X_train, axis=0)
        protected_feat = {
            'Payment Format_target_enc', 'Receiving Currency_target_enc', 'Payment Currency_target_enc',
            'is_toxic_corridor', 'flag_heavy_structuring', 'prev_was_sender'
        }
        variance_mask = []
        for f, var in zip(features, variances):
            if f in protected_feat:
                variance_mask.append(True)
            else:
                variance_mask.append(var > 0.001)
        
        variance_mask = np.array(variance_mask)
        removed_feats = [f for f, keep in zip(features, variance_mask) if not keep]
        if removed_feats:
            logger.warning(f"Removed {len(removed_feats)} low-variance features: {removed_feats}")
        X_train = X_train[:, variance_mask]
        X_val = X_val[:,   variance_mask]
        features = [f for f, keep in zip(features, variance_mask) if keep]
        logger.info(f"Kept {len(features)} features after variance filter")
 
        n_fraud = int(y_train.sum())
        imbalance_ratio = (len(y_train) - n_fraud) / max(n_fraud, 1)
        logger.info(f"Class imbalance: 1:{imbalance_ratio:.0f}")
 
        #split val into cal and eval slices
        mid= len(X_val) // 2
        X_cal,  y_cal = X_val[:mid], y_val[:mid]
        X_eval, y_eval = X_val[mid:], y_val[mid:]
 
        has_anomaly = "anomaly_score" in features
        if has_anomaly:
            a_idx = features.index("anomaly_score")
            anomaly_cal = X_cal[:, a_idx].copy()
            anomaly_eval = X_eval[:, a_idx].copy()
        else:
            anomaly_cal = anomaly_eval = None
 
        #MCCV model selection
        logger.info("=" * 60)
        logger.info("MCCV Model Selection")
        logger.info("=" * 60)
 
        model_configs = get_model_configs(imbalance_ratio, random_state)
        mccv_results = {}
        all_results = []
 
        for name, config in model_configs.items():
            try:
                logger.info(f"Evaluating {name}...")
                result = mccv_evaluate(
                    config, X_train, y_train, train_ts,
                    n_folds=n_folds, random_state=random_state,
                    fpr_threshold=fpr_threshold
                )
                mccv_results[name] = result
                all_results.append({"name": name, "mccv": result})
                logger.info(
                    f" {result['mean_recall']:.4f} +- {result['std_recall']:.4f} "
                    f"(conservative: {result['conservative_score']:.4f})"
                )
            except Exception as e:
                logger.error(f"{name} failed: {e}")
 
        if not mccv_results:
            raise RuntimeError("All models failed MCCV, cannot continue")
 
        best_name, best_mccv = select_best_model(mccv_results)
        best_config = model_configs[best_name]
 
        #final training 
        logger.info("=" * 60)
        logger.info("Final Training")
        logger.info("=" * 60)
 
        final_model = build_model(best_config)
        final_model.fit(X_train, y_train)
 
        raw_probs_eval = final_model.predict_proba(X_eval)[:, 1]
        val_metrics = calculate_metrics(y_eval, raw_probs_eval, fpr_threshold)
        logger.info(
            f"Val — Recall@1%FPR: {val_metrics['recall_at_1pct_fpr']:.4f} | "
            f"PR-AUC: {val_metrics['pr_auc']:.4f}"
        )
 
        #MCCV vs val alignment check
        gap = abs(best_mccv["mean_recall"] - val_metrics["recall_at_1pct_fpr"])
        if gap > 0.05:
            logger.warning(
                f"{best_name}: MCCV recall {best_mccv['mean_recall']:.4f} vs "
                f"val recall {val_metrics['recall_at_1pct_fpr']:.4f} — gap={gap:.4f}"
            )
        else:
            logger.info(f"MCCV/val aligned — gap={gap:.4f}")
 
        # calibration
        platt= fit_platt_scaler(final_model, X_cal, y_cal)
        probs_eval = calibrated_predict(final_model, platt, X_eval)
        cal_error = check_calibration(y_eval, probs_eval, best_name, shap_dir)
        opt_threshold = find_optimal_threshold(y_eval, probs_eval, fpr_threshold)
 
        #test evaluation 
        logger.info("=" *60)
        logger.info("Test Evaluation")
        logger.info("=" * 60)
 
        X_test, y_test, _ = load_split(
            test_path, features, target_col, time_col,
            max_rows=model_cfg.get("test_sample_size", 750_000),
            random_state=random_state
        )
        X_test= X_test[:, variance_mask] if X_test.shape[1] > len(features) else X_test
        test_probs = calibrated_predict(final_model, platt, X_test)
        test_metrics = calculate_metrics(y_test, test_probs, fpr_threshold)

        #save model predictions for threshold tunuing
        artifacts_dir = model_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        np.save(artifacts_dir / "y_true.npy", y_test)
        np.save(artifacts_dir / "y_prob.npy", test_probs)

        logger.info(f"Saved predictions to {artifacts_dir}")
        logger.info(
            f"Test — Recall@1%FPR: {test_metrics['recall_at_1pct_fpr']:.4f} | "
            f"PR-AUC: {test_metrics['pr_auc']:.4f} | "
            f"ROC-AUC: {test_metrics['roc_auc']:.4f}"
        )
 
        plot_thresholds(y_test, test_probs, best_name, fpr_threshold, shap_dir)
 
        #SHAP on test sample
        _, shap_kept = explain_and_select_features(
            final_model, X_test[:1000], features, best_name, shap_dir
        )
 
        del X_test, X_train, X_val, X_cal, X_eval
        gc.collect()
 
        #save model artifact
        model_path = model_dir / f"{best_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":final_model,
                "platt": platt,
                "features":features,
                "optimal_threshold": opt_threshold,
                "shap_kept_features": shap_kept,
                "variance_mask": variance_mask,
                "imbalance_ratio": imbalance_ratio,
                "mccv_results":best_mccv,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "calibration_error": cal_error,
                "best_model_name":best_name,
            }, f)
        logger.info(f"Model saved -> {model_path}")
 
        #save metrics for DVC + model_evaluation 
        all_metrics = {
            "best_model":best_name,
            "mccv_mean_recall": best_mccv["mean_recall"],
            "mccv_std_recall": best_mccv["std_recall"],
            "mccv_conservative": best_mccv["conservative_score"],
            "mccv_val_gap": gap,
            "val_recall_at_1pct_fpr": val_metrics["recall_at_1pct_fpr"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "test_recall_at_1pct_fpr": test_metrics["recall_at_1pct_fpr"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_roc_auc":test_metrics["roc_auc"],
            "calibration_error":cal_error,
            "optimal_threshold": opt_threshold,
            "n_features": len(features),
        }
 
        metrics_path = reports_dir / "model_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Metrics saved -> {metrics_path}")
 
        logger.info("=" * 60)
        logger.info("Model building complete")
        logger.info(f"Best model: {best_name}")
        logger.info(f"Test AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Test Recall@1%FPR: {test_metrics['recall_at_1pct_fpr']:.4f}")
        logger.info("=" * 60)
 
    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise
 
 
if __name__ == "__main__":
    main()