"""
Model evaluation module
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Class for evaluating spam detection models"""

    def __init__(self, config: dict):
        """
        Initialize ModelEvaluator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})
        self.report_path = self.eval_config.get('report_path', 'reports/')

        # Create report directory
        Path(self.report_path).mkdir(parents=True, exist_ok=True)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating evaluation metrics")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary')
        }

        # Add AUC if probabilities provided
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        logger.info("Metrics calculated:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ) -> None:
        """
        Plot and save confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot (optional)
        """
        logger.info("Creating confusion matrix plot")

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path is None:
            save_path = Path(self.report_path) / 'confusion_matrix.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = None
    ) -> None:
        """
        Plot and save ROC curve

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot (optional)
        """
        logger.info("Creating ROC curve plot")

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path is None:
            save_path = Path(self.report_path) / 'roc_curve.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
        plt.close()

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list,
        save_path: str = None
    ) -> None:
        """
        Plot feature importance for tree-based models

        Args:
            model: Trained model
            feature_names: List of feature names
            save_path: Path to save plot (optional)
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return

        logger.info("Creating feature importance plot")

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance')
        plt.ylabel('Importance')
        plt.xlabel('Features')
        plt.tight_layout()

        if save_path is None:
            save_path = Path(self.report_path) / 'feature_importance.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ) -> str:
        """
        Generate and save classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save report (optional)

        Returns:
            Classification report as string
        """
        logger.info("Generating classification report")

        report = classification_report(
            y_true,
            y_pred,
            target_names=['Ham', 'Spam'],
            digits=4
        )

        logger.info(f"\n{report}")

        if save_path is None:
            save_path = Path(self.report_path) / 'classification_report.txt'

        with open(save_path, 'w') as f:
            f.write(report)

        logger.info(f"Classification report saved to {save_path}")

        return report

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list = None
    ) -> Dict[str, float]:
        """
        Complete model evaluation with all metrics and plots

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names (optional)

        Returns:
            Dictionary of metrics
        """
        logger.info("Starting comprehensive model evaluation")

        # Make predictions
        y_pred = model.predict(X_test)

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Generate plots
        if self.eval_config.get('save_confusion_matrix', True):
            self.plot_confusion_matrix(y_test, y_pred)

        if self.eval_config.get('save_roc_curve', True) and y_pred_proba is not None:
            self.plot_roc_curve(y_test, y_pred_proba)

        if feature_names is not None:
            self.plot_feature_importance(model, feature_names)

        # Generate classification report
        if self.eval_config.get('save_classification_report', True):
            self.generate_classification_report(y_test, y_pred)

        logger.info("Model evaluation complete")

        return metrics
