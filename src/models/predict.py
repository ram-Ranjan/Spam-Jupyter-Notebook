"""
Model prediction/inference module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import joblib
from typing import Union, List, Dict

from src.utils.logger import get_logger
from src.features.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class SpamPredictor:
    """Class for making predictions on new messages"""

    def __init__(self, config: dict):
        """
        Initialize SpamPredictor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_engineer = None
        self.model_loaded = False

    def load_artifacts(
        self,
        model_path: str = None,
        scaler_path: str = None,
        vectorizer_path: str = None
    ) -> None:
        """
        Load model and preprocessing artifacts

        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            vectorizer_path: Path to vectorizer file
        """
        logger.info("Loading model artifacts")

        model_save_config = self.config.get('model_save', {})

        # Default paths
        if model_path is None:
            model_path = model_save_config.get('model_path', 'models/spam_classifier.pkl')
        if scaler_path is None:
            scaler_path = model_save_config.get('scaler_path', 'models/scaler.pkl')
        if vectorizer_path is None:
            vectorizer_path = model_save_config.get('vectorizer_path', 'models/tfidf_vectorizer.pkl')

        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load scaler
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.config)

        # Load vectorizer if exists
        if Path(vectorizer_path).exists():
            self.feature_engineer.vectorizer = joblib.load(vectorizer_path)
            logger.info(f"Vectorizer loaded from {vectorizer_path}")

        self.model_loaded = True

    def predict_single(self, message: str) -> Dict[str, Union[str, float]]:
        """
        Predict if a single message is spam or ham

        Args:
            message: Text message to classify

        Returns:
            Dictionary with prediction and confidence
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_artifacts first.")

        # Create dataframe with single message
        df = pd.DataFrame({'Message': [message]})

        # Extract features
        features = self.feature_engineer.extract_all_features(df, fit_tfidf=False)

        # Scale if scaler exists
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Make prediction
        prediction = self.model.predict(features)[0]
        prediction_label = 'Spam' if prediction == 1 else 'Ham'

        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            confidence = float(proba[1]) if prediction == 1 else float(proba[0])
        else:
            confidence = None

        result = {
            'message': message,
            'prediction': prediction_label,
            'prediction_code': int(prediction),
            'confidence': confidence
        }

        logger.info(f"Prediction: {prediction_label} (confidence: {confidence:.4f if confidence else 'N/A'})")

        return result

    def predict_batch(self, messages: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict multiple messages

        Args:
            messages: List of text messages

        Returns:
            List of prediction dictionaries
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_artifacts first.")

        logger.info(f"Predicting {len(messages)} messages")

        results = []
        for message in messages:
            result = self.predict_single(message)
            results.append(result)

        return results

    def predict_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Predict messages from a CSV file

        Args:
            file_path: Path to CSV file with 'Message' column

        Returns:
            DataFrame with predictions
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_artifacts first.")

        logger.info(f"Predicting messages from {file_path}")

        # Load data
        df = pd.read_csv(file_path, encoding='utf-8')

        if 'Message' not in df.columns:
            raise ValueError("CSV file must have 'Message' column")

        # Extract features
        features = self.feature_engineer.extract_all_features(df, fit_tfidf=False)

        # Scale if scaler exists
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        # Make predictions
        predictions = self.model.predict(features_scaled)
        df['prediction'] = ['Spam' if p == 1 else 'Ham' for p in predictions]
        df['prediction_code'] = predictions

        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(features_scaled)
            df['confidence'] = [probas[i, p] for i, p in enumerate(predictions)]

        logger.info(f"Predictions complete: {(predictions == 1).sum()} spam, {(predictions == 0).sum()} ham")

        return df
