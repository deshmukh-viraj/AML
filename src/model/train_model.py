"""
AML Model Training with XGBoost, Class Weights, and Explainability

This module handles:
1. XGBoost classification with scale_pos_weight for class imbalance
2. Hyperparameter optimization targeting recall/PR-AUC
3. Threshold tuning for operational metrics (precision, recall, F2)
4. SHAP explainability for model interpretation
5. Model evaluation and metrics computation
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Dict, Tuple, Optional

import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score,
    classification_report,
    matthews_corrcoef, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AMLXGBoostModel:
    """
    XGBoost-based AML classification model with advanced training strategies.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.05,
                 max_depth: int = 6,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 objective: str = 'binary:logistic',
                 eval_metric: str = 'aucpr',
                 seed: int = 42,
                 **kwargs):  # Accept additional parameters for model loading
        """Initialize XGBoost parameters."""
        
        self.params = {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': objective,
            'eval_metric': eval_metric,
            'seed': seed,
            'tree_method': kwargs.get('tree_method', 'hist'),
            'device': kwargs.get('device', 'cuda'),  # Use GPU if available
            'verbosity': kwargs.get('verbosity', 1),
        }
        
        self.model: Optional[xgb.Booster] = None
        self.feature_names: list = []
        self.threshold: float = 0.5  # Default decision threshold
        self.explainer: Optional[shap.TreeExplainer] = None
        
    def prepare_data(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.
        """
        pdf = df.to_pandas()
        
        # Drop non-feature columns
        exclude_cols = [
            'Is_Laundering', 'Is Laundering',
            'Account_HASHED', 'Account_duplicated_0', 'Timestamp',
            'split', 'account_pair_directed', 'account_pair_undirected',
            'From Bank', 'To Bank', 'Payment Currency', 'Payment Format',
            'hour_window', 'day_window', 'day_of_week', 'hour_of_day'
        ]
        
        feature_cols = [col for col in pdf.columns 
                       if col not in exclude_cols
                       and pdf[col].dtype in ['float64', 'int64']
                       and not pdf[col].isna().all()]
        
        self.feature_names = feature_cols
        
        X = pdf[feature_cols].fillna(pdf[feature_cols].median())
        
        # Handle target variable (account for column name variations)
        target_col = 'Is_Laundering' if 'Is_Laundering' in pdf.columns else 'Is Laundering'
        y = pdf[target_col].values.astype(int)
        
        logger.info(f"Features selected: {len(feature_cols)}")
        logger.info(f"Target distribution: {(y == 0).sum()} negative, {(y == 1).sum()} positive")
        logger.info(f"Class imbalance ratio: {(y == 0).sum() / ((y == 1).sum() + 1e-6):.2f}:1")
        
        return X, y
    
    def compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Compute class weight for imbalanced dataset.
        
        scale_pos_weight = (negative samples) / (positive samples)
        Higher weight = more penalty for false negatives (recall optimized)
        """
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        
        scale_pos_weight = n_neg / max(n_pos, 1)
        logger.info(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")
        
        return scale_pos_weight
    
    def train(self, 
              train_df: pl.DataFrame,
              val_df: Optional[pl.DataFrame] = None,
              num_rounds: int = 500):
        """
        Train XGBoost with class weights and early stopping.
        
        Args:
            train_df: Training data (Polars DataFrame)
            val_df: Validation data for early stopping
            num_rounds: Maximum boosting rounds
        """
        logger.info("="*60)
        logger.info("TRAINING XGBoost MODEL")
        logger.info("="*60)
        
        # Prepare training data
        X_train, y_train = self.prepare_data(train_df)
        
        # Compute and apply class weight
        scale_pos_weight = self.compute_scale_pos_weight(y_train)
        self.params['scale_pos_weight'] = scale_pos_weight
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Setup early stopping with validation set
        evals = None
        evals_result = {}
        
        if val_df is not None:
            X_val, y_val = self.prepare_data(val_df)
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
            
            # Early stopping callback
            def early_stopping_callback(env):
                """Stop if val metric doesn't improve for 50 rounds."""
                current = env.evaluation_result_list[-1][1]
                if not hasattr(early_stopping_callback, 'best'):
                    early_stopping_callback.best = current
                    early_stopping_callback.rounds_without_improvement = 0
                elif current > early_stopping_callback.best:  # aucpr higher is better
                    early_stopping_callback.best = current
                    early_stopping_callback.rounds_without_improvement = 0
                else:
                    early_stopping_callback.rounds_without_improvement += 1
                
                if early_stopping_callback.rounds_without_improvement > 50:
                    return True
        
        # Train
        logger.info(f"Training for up to {num_rounds} rounds...")
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=50,
            callbacks=[
                xgb.callback.EarlyStopping(rounds=50, metric_name='aucpr', save_best=True)
            ] if val_df is not None else None
        )
        
        logger.info(f"✓ Training complete. Best iteration: {self.model.best_iteration}")
    
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call .train() first.")
        
        pdf = df.to_pandas()
        X = pdf[self.feature_names].fillna(pdf[self.feature_names].median())
        
        dmatrix = xgb.DMatrix(X)
        probas = self.model.predict(dmatrix)
        
        return probas
    
    def predict_with_threshold(self, df: pl.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary predictions using custom threshold.
        """
        probas = self.predict(df)
        return (probas >= threshold).astype(int)
    
    def tune_threshold(self,
                      val_df: pl.DataFrame,
                      target_metric: str = 'f2',
                      beta: float = 2.0) -> float:
        """
        Tune decision threshold to optimize for desired metric.
        
        Args:
            val_df: Validation set for threshold tuning
            target_metric: 'recall', 'precision', 'f1', 'f2' (recall-focused)
            beta: Beta parameter for F-beta score (higher = favor recall)
        
        Returns:
            Optimal threshold
        """
        logger.info("="*60)
        logger.info("THRESHOLD TUNING")
        logger.info("="*60)
        
        probas = self.predict(val_df)
        pdf = val_df.to_pandas()
        target_col = 'Is_Laundering' if 'Is_Laundering' in pdf.columns else 'Is Laundering'
        y_true = pdf[target_col].values
        
        # Compute metrics across thresholds
        thresholds = np.arange(0.1, 0.95, 0.01)
        metrics = {
            'threshold': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'f2': [],
        }
        
        for threshold in thresholds:
            y_pred = (probas >= threshold).astype(int)
            
            metrics['threshold'].append(threshold)
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
            
            # F-beta score
            if metrics['recall'][-1] + metrics['precision'][-1] > 0:
                f_beta = (1 + beta**2) * (metrics['precision'][-1] * metrics['recall'][-1]) / \
                         ((beta**2 * metrics['precision'][-1]) + metrics['recall'][-1] + 1e-10)
                metrics['f2'].append(f_beta)
            else:
                metrics['f2'].append(0)
        
        # Select threshold based on target metric
        metrics_df = pd.DataFrame(metrics)
        
        if target_metric == 'f2':
            best_idx = metrics_df['f2'].idxmax()
        elif target_metric == 'f1':
            best_idx = metrics_df['f1'].idxmax()
        elif target_metric == 'recall':
            # Target high recall, but minimum precision > 0.5
            filtered = metrics_df[metrics_df['precision'] >= 0.5]
            best_idx = filtered['recall'].idxmax() if len(filtered) > 0 else 0
        elif target_metric == 'precision':
            best_idx = metrics_df['precision'].idxmax()
        else:
            best_idx = metrics_df['f2'].idxmax()
        
        self.threshold = metrics_df.loc[best_idx, 'threshold']
        
        logger.info(f"\nOptimal threshold: {self.threshold:.3f}")
        logger.info(f"  Precision: {metrics_df.loc[best_idx, 'precision']:.4f}")
        logger.info(f"  Recall: {metrics_df.loc[best_idx, 'recall']:.4f}")
        logger.info(f"  F1: {metrics_df.loc[best_idx, 'f1']:.4f}")
        logger.info(f"  F2: {metrics_df.loc[best_idx, 'f2']:.4f}")
        
        return self.threshold
    
    def evaluate(self, df: pl.DataFrame, threshold: Optional[float] = None) -> Dict:
        """
        Comprehensive model evaluation.
        """
        if threshold is None:
            threshold = self.threshold
        
        probas = self.predict(df)
        y_pred = (probas >= threshold).astype(int)
        
        pdf = df.to_pandas()
        target_col = 'Is_Laundering' if 'Is_Laundering' in pdf.columns else 'Is Laundering'
        y_true = pdf[target_col].values
        
        # Compute metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Compute PR-AUC with proper sorting
        prec, rec, _ = precision_recall_curve(y_true, probas)
        # Sort by recall (ascending) to ensure monotonic increasing order
        sorted_indices = np.argsort(rec)
        rec_sorted = rec[sorted_indices]
        prec_sorted = prec[sorted_indices]
        pr_auc_val = auc(rec_sorted, prec_sorted)
        
        metrics = {
            'threshold': threshold,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, probas),
            'pr_auc': pr_auc_val,
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }
        
        return metrics
    
    def explain_predictions(self,
                           df: pl.DataFrame,
                           sample_size: int = 100,
                           top_k: int = 15):
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            df: Data to explain
            sample_size: Number of background samples for SHAP
            top_k: Number of top features to show
        """
        logger.info("="*60)
        logger.info("GENERATING SHAP EXPLANATIONS")
        logger.info("="*60)
        
        pdf = df.to_pandas()
        X = pdf[self.feature_names].fillna(pdf[self.feature_names].median())
        
        # Create SHAP explainer
        dmatrix = xgb.DMatrix(X)
        
        # Use sample for efficiency
        sample_idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
        X_sample = X.iloc[sample_idx]
        dmatrix_sample = xgb.DMatrix(X_sample)
        
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(dmatrix_sample)
        
        logger.info(f"Generated SHAP values for {len(X_sample)} samples")
        
        # Global feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
        }).sort_values('mean_abs_shap', ascending=False)
        
        logger.info("\nTop 10 Most Important Features (by SHAP):")
        for idx, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return shap_values, X_sample, importance
    
    def save(self, path: Path):
        """Save trained model."""
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'params': self.params,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved to {path}")
    
    @staticmethod
    def load(path: Path) -> 'AMLXGBoostModel':
        """Load trained model."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = AMLXGBoostModel(**model_data['params'])
        model.model = model_data['model']
        model.feature_names = model_data['feature_names']
        model.threshold = model_data['threshold']
        
        logger.info(f"✓ Model loaded from {path}")
        return model


def train_aml_model(train_df: pl.DataFrame,
                   val_df: pl.DataFrame,
                   test_df: pl.DataFrame,
                   model_output_path: Path) -> AMLXGBoostModel:
    """
    End-to-end training pipeline.
    """
    logger.info("\n" + "="*60)
    logger.info("AML XGBoost Training Pipeline")
    logger.info("="*60)
    
    # Initialize model
    model = AMLXGBoostModel(
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Train
    model.train(train_df, val_df, num_rounds=500)
    
    # Evaluate on validation
    val_metrics = model.evaluate(val_df)
    logger.info(f"\nValidation Metrics (default threshold 0.5):")
    for key, value in val_metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Tune threshold for recall optimization (F2-score)
    model.tune_threshold(val_df, target_metric='f2', beta=2.0)
    
    # Evaluate with tuned threshold
    val_metrics_tuned = model.evaluate(val_df, threshold=model.threshold)
    logger.info(f"\nValidation Metrics (tuned threshold {model.threshold:.3f}):")
    for key, value in val_metrics_tuned.items():
        logger.info(f"  {key}: {value}")
    
    # Test set evaluation
    test_metrics = model.evaluate(test_df, threshold=model.threshold)
    logger.info(f"\nTest Metrics (tuned threshold {model.threshold:.3f}):")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Generate explanations
    shap_values, X_sample, feature_importance = model.explain_predictions(val_df, sample_size=100)
    
    # Save model
    model.save(model_output_path)
    
    return model
