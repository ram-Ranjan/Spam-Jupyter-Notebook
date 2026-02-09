"""
Unit tests for data_loader module
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'data': {
            'raw_path': 'data/raw/Spam_or_ham.csv',
            'encoding': 'utf-8'
        }
    }


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing"""
    return pd.DataFrame({
        'Message': [
            'नमस्ते, कैसे हैं आप?',
            'फ्री में पाएं 10000 रूपए',
            'धन्यवाद'
        ],
        'Category': ['ham', 'spam', 'ham']
    })


class TestDataLoader:
    """Test cases for DataLoader class"""

    def test_init(self, sample_config):
        """Test DataLoader initialization"""
        loader = DataLoader(sample_config)
        assert loader.config == sample_config
        assert loader.data_config == sample_config['data']

    def test_prepare_target(self, sample_config, sample_dataframe):
        """Test target variable preparation"""
        loader = DataLoader(sample_config)
        df = loader.prepare_target(sample_dataframe)

        assert 'is_spam' in df.columns
        assert df['is_spam'].dtype == 'int64'
        assert df['is_spam'].sum() == 1  # One spam message
        assert len(df[df['is_spam'] == 0]) == 2  # Two ham messages

    def test_validate_data_success(self, sample_config, sample_dataframe):
        """Test data validation with valid data"""
        loader = DataLoader(sample_config)
        # Should not raise any exception
        loader._validate_data(sample_dataframe)

    def test_validate_data_missing_column(self, sample_config):
        """Test data validation with missing columns"""
        loader = DataLoader(sample_config)
        invalid_df = pd.DataFrame({'Message': ['test']})

        with pytest.raises(ValueError, match="Missing required columns"):
            loader._validate_data(invalid_df)

    def test_get_data_summary(self, sample_config, sample_dataframe):
        """Test data summary generation"""
        loader = DataLoader(sample_config)
        df = loader.prepare_target(sample_dataframe)
        summary = loader.get_data_summary(df)

        assert summary['total_messages'] == 3
        assert summary['spam_count'] == 1
        assert summary['ham_count'] == 2
        assert 'Message' in summary['columns']
        assert 'Category' in summary['columns']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
