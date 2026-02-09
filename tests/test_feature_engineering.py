"""
Unit tests for feature_engineering module
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'features': {
            'tfidf_max_features': 10,
            'spam_words': ['फ्री', 'मुफ्त', 'लॉटरी'],
            'patterns': {
                'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                'phone': r'\b[\+]?[0-9][0-9\-]{9,}\b',
                'money': r'(?:₹|RS|INR|\$)\s*\d+(?:,\d+)*(?:\.\d{2})?',
                'special_chars': r'[!@#$%^&*(),.?":{}|<>]'
            }
        }
    }


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing"""
    return pd.DataFrame({
        'Message': [
            'नमस्ते, कैसे हैं आप?',
            'फ्री में पाएं ₹10000',
            'कॉल करें 9876543210'
        ]
    })


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""

    def test_init(self, sample_config):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer(sample_config)
        assert engineer.config == sample_config
        assert engineer.spam_words == ['फ्री', 'मुफ्त', 'लॉटरी']
        assert engineer.vectorizer is None

    def test_extract_basic_features(self, sample_config):
        """Test basic feature extraction"""
        engineer = FeatureEngineer(sample_config)
        message = "नमस्ते दोस्त"
        features = engineer.extract_basic_features(message)

        assert 'message_length' in features
        assert 'word_count' in features
        assert features['message_length'] == len(message)
        assert features['word_count'] == 2

    def test_extract_pattern_features_with_url(self, sample_config):
        """Test pattern feature extraction with URL"""
        engineer = FeatureEngineer(sample_config)
        message = "Visit https://example.com for details"
        features = engineer.extract_pattern_features(message)

        assert features['has_url'] == 1
        assert 'has_phone' in features
        assert 'has_money' in features
        assert 'special_char_count' in features

    def test_extract_pattern_features_with_phone(self, sample_config):
        """Test pattern feature extraction with phone"""
        engineer = FeatureEngineer(sample_config)
        message = "कॉल करें 9876543210"
        features = engineer.extract_pattern_features(message)

        assert features['has_phone'] == 1

    def test_extract_pattern_features_with_money(self, sample_config):
        """Test pattern feature extraction with money"""
        engineer = FeatureEngineer(sample_config)
        message = "केवल ₹1000 में"
        features = engineer.extract_pattern_features(message)

        assert features['has_money'] == 1

    def test_count_spam_words(self, sample_config):
        """Test spam word counting"""
        engineer = FeatureEngineer(sample_config)
        message = "फ्री में पाएं मुफ्त लॉटरी"
        count = engineer.count_spam_words(message)

        assert count == 3  # All three spam words present

    def test_count_spam_words_none(self, sample_config):
        """Test spam word counting with no spam words"""
        engineer = FeatureEngineer(sample_config)
        message = "नमस्ते कैसे हैं आप"
        count = engineer.count_spam_words(message)

        assert count == 0

    def test_extract_all_features(self, sample_config, sample_dataframe):
        """Test complete feature extraction"""
        engineer = FeatureEngineer(sample_config)
        features_df = engineer.extract_all_features(sample_dataframe, fit_tfidf=True)

        assert len(features_df) == len(sample_dataframe)
        assert 'message_length' in features_df.columns
        assert 'word_count' in features_df.columns
        assert 'has_url' in features_df.columns
        assert 'has_phone' in features_df.columns
        assert 'has_money' in features_df.columns
        assert 'special_char_count' in features_df.columns
        assert 'spam_word_count' in features_df.columns
        assert 'tfidf_score' in features_df.columns
        assert engineer.vectorizer is not None

    def test_get_feature_names(self, sample_config):
        """Test feature names retrieval"""
        engineer = FeatureEngineer(sample_config)
        feature_names = engineer.get_feature_names()

        assert len(feature_names) == 8
        assert 'message_length' in feature_names
        assert 'tfidf_score' in feature_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
