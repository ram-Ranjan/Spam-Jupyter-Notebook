"""
Model training module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from typing import Tuple, Dict, Any
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Class for training spam detection models"""

    def __init__(self, config: dict):
        """
        Initialize ModelTrainer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.model = None
        self.scaler = None

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets

        Args:
            X: Features dataframe
            y: Target series

        Returns:
            X_train, X_test, y_train, y_test
        """
        test_size = self.model_config.get('test_size', 0.2)
        random_state = self.model_config.get('random_state_split', 100)

        logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def preprocess_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame = None
    ) -> Tuple[np.ndarray, pd.Series, np.ndarray]:
        """
        Preprocess data: scale and optionally oversample

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)

        Returns:
            X_train_processed, y_train_processed, X_test_processed
        """
        logger.info("Preprocessing data")

        # Scale features
        scale_features = self.training_config.get('scale_features', True)
        if scale_features:
            logger.info("Scaling features")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values if X_test is not None else None

        # Oversample if needed
        oversample = self.training_config.get('oversample', False)
        if oversample:
            logger.info("Oversampling minority class")
            ros = RandomOverSampler(random_state=42)
            X_train_scaled, y_train = ros.fit_resample(X_train_scaled, y_train)

            spam_count = y_train.sum()
            ham_count = len(y_train) - spam_count
            logger.info(f"After oversampling - Spam: {spam_count}, Ham: {ham_count}")

        return X_train_scaled, y_train, X_test_scaled

    def get_model(self, model_name: str = None) -> Any:
        """
        Get model instance based on configuration

        Args:
            model_name: Model name (if None, uses config)

        Returns:
            Model instance
        """
        if model_name is None:
            model_name = self.model_config.get('name', 'RandomForest')

        logger.info(f"Creating model: {model_name}")

        if model_name == 'RandomForest':
            rf_params = self.model_config.get('rf_params', {})
            model = RandomForestClassifier(
                random_state=self.model_config.get('random_state', 45),
                **rf_params
            )
        elif model_name == 'LogisticRegression':
            lr_params = self.model_config.get('lr_params', {})
            model = LogisticRegression(**lr_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        model_name: str = None
    ) -> Any:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training target
            model_name: Model name (optional)

        Returns:
            Trained model
        """
        logger.info("Starting model training")

        self.model = self.get_model(model_name)
        self.model.fit(X_train, y_train)

        logger.info("Model training complete")

        # Cross-validation if enabled
        if self.training_config.get('cross_validation', False):
            cv_folds = self.training_config.get('cv_folds', 5)
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=cv_folds, scoring='accuracy'
            )
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.model

    def save_model(self, save_dir: str = None) -> None:
        """
        Save trained model and scaler

        Args:
            save_dir: Directory to save models (if None, uses config)
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        logger.info("Saving model artifacts")

        # Get save paths
        if save_dir:
            model_path = Path(save_dir) / 'spam_classifier.pkl'
            scaler_path = Path(save_dir) / 'scaler.pkl'
        else:
            model_path = self.config.get('model_save', {}).get('model_path', 'models/spam_classifier.pkl')
            scaler_path = self.config.get('model_save', {}).get('scaler_path', 'models/scaler.pkl')

        # Create directory if needed
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save scaler if it exists
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path: str = None, scaler_path: str = None) -> None:
        """
        Load trained model and scaler

        Args:
            model_path: Path to model file (if None, uses config)
            scaler_path: Path to scaler file (if None, uses config)
        """
        logger.info("Loading model artifacts")

        # Get load paths
        if model_path is None:
            model_path = self.config.get('model_save', {}).get('model_path', 'models/spam_classifier.pkl')
        if scaler_path is None:
            scaler_path = self.config.get('model_save', {}).get('scaler_path', 'models/scaler.pkl')

        # Load model
        if Path(model_path).exists():
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load scaler if exists
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
