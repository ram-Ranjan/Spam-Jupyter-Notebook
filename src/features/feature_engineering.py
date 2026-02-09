"""
Feature engineering module for spam detection
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Class for extracting features from text messages"""

    def __init__(self, config: dict):
        """
        Initialize FeatureEngineer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config.get('features', {})
        self.patterns = self.feature_config.get('patterns', {})
        self.spam_words = self.feature_config.get('spam_words', [])
        self.vectorizer = None
        self.tfidf_scores = None

    def extract_basic_features(self, message: str) -> Dict[str, int]:
        """
        Extract basic features from message

        Args:
            message: Text message

        Returns:
            Dictionary of basic features
        """
        return {
            'message_length': len(message),
            'word_count': len(message.split())
        }

    def extract_pattern_features(self, message: str) -> Dict[str, int]:
        """
        Extract pattern-based features from message

        Args:
            message: Text message

        Returns:
            Dictionary of pattern features
        """
        return {
            'has_url': int(bool(re.search(self.patterns.get('url', ''), message))),
            'has_phone': int(bool(re.search(self.patterns.get('phone', ''), message))),
            'has_money': int(bool(re.search(self.patterns.get('money', ''), message))),
            'special_char_count': len(re.findall(self.patterns.get('special_chars', ''), message))
        }

    def count_spam_words(self, message: str) -> int:
        """
        Count spam-related keywords in message

        Args:
            message: Text message

        Returns:
            Count of spam words
        """
        return sum(word in message for word in self.spam_words)

    def fit_tfidf(self, messages: pd.Series) -> None:
        """
        Fit TF-IDF vectorizer on messages

        Args:
            messages: Series of text messages
        """
        logger.info("Fitting TF-IDF vectorizer")

        max_features = self.feature_config.get('tfidf_max_features', 100)
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = self.vectorizer.fit_transform(messages)
        self.tfidf_scores = tfidf_matrix.mean(axis=1).A1

        logger.info(f"TF-IDF fitted with {max_features} features")

    def transform_tfidf(self, messages: pd.Series) -> np.ndarray:
        """
        Transform messages using fitted TF-IDF vectorizer

        Args:
            messages: Series of text messages

        Returns:
            TF-IDF scores

        Raises:
            ValueError: If vectorizer not fitted
        """
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")

        tfidf_matrix = self.vectorizer.transform(messages)
        return tfidf_matrix.mean(axis=1).A1

    def extract_all_features(self, df: pd.DataFrame, fit_tfidf: bool = True) -> pd.DataFrame:
        """
        Extract all features from dataframe

        Args:
            df: Input dataframe with 'Message' column
            fit_tfidf: Whether to fit TF-IDF (True for training, False for prediction)

        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting features from messages")

        # Fit TF-IDF if needed
        if fit_tfidf:
            self.fit_tfidf(df['Message'])
            tfidf_scores = self.tfidf_scores
        else:
            tfidf_scores = self.transform_tfidf(df['Message'])

        # Extract features for each message
        features_list = []
        for idx, message in enumerate(df['Message']):
            features = {}
            features.update(self.extract_basic_features(message))
            features.update(self.extract_pattern_features(message))
            features['spam_word_count'] = self.count_spam_words(message)
            features['tfidf_score'] = tfidf_scores[idx]
            features_list.append(features)

        # Create features dataframe
        features_df = pd.DataFrame(features_list)

        logger.info(f"Feature extraction complete. Shape: {features_df.shape}")
        logger.info(f"Features: {list(features_df.columns)}")

        return features_df

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names

        Returns:
            List of feature names
        """
        return [
            'message_length',
            'word_count',
            'has_url',
            'has_phone',
            'has_money',
            'special_char_count',
            'spam_word_count',
            'tfidf_score'
        ]
