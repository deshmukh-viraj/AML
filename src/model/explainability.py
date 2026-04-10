import shap
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AMLExplainer:
    def __init__(self, model, feature_names: List[str]):
        """initialize SHAP explainer"""
        self.model = model
        self.feature_names = feature_names

        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            self.explainer = shap.TreeExplainer(model.named_steps['clf'])
        else:
            self.explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized")

    
    def get_global_importance(self, X_sample: np.ndarray, save_dir: str) -> None:
        """
        generates globla SHAP plots and csv for mlflow logging"""

        logger.info(f"Generating global SHAP explanations for {X_sample.shape[0]} sample...")

        #calculate shap values
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10,8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/xai_shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10,8))
        shap.summary_plot(shap_values,X_sample, feature_names=self.feature_names, show=False, max_display=20, plot_type='bar')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/xai_shap_importance.png", dpi=150, bbox_inches='tight')
        plt.close()

        #machine readable csv
        mean_shap = np.abs(shap_values).mean(axis=0)
        imp_df = pl.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_shap
        }).sort(by='mean_abs_shap', descending=True)

        csv_path = f"{save_dir}/xai_feature_importance.csv"
        imp_df.write_csv(csv_path)
        logger.info(f"Saved XAI artifacts to {save_dir}")

    
    def explain_transactions(self, feature_dict: Dict[str, Any])-> List[Dict[str, float]]:
        """
        generates local SHAP values for a single transaction 
        """
        X_input = np.array([[feature_dict.get(f, 0.0) for f in self.feature_names]], dtype=np.float32)
        shap_values = self.explainer.shap_values(X_input)[0]

        explanation = [
            {
                "feature": self.feature_names[i],
                "value": float(X_input[0][i]),
                "shap_values": float(shap_values[i])
            }
            for i in np.argsort(np.abs(shap_values))[::-1][:5]
        ]
        return explanation