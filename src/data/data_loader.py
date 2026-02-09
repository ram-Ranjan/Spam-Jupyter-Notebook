"""
Data loading and preprocessing module
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Class for loading and validating spam detection data"""

    def __init__(self, config: dict):
        """
        Initialize DataLoader

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            file_path: Path to CSV file (if None, uses config path)

        Returns:
            Loaded dataframe

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        if file_path is None:
            file_path = self.data_config.get('raw_path')

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        # Load with appropriate encoding
        encoding = self.data_config.get('encoding', 'utf-8')
        df = pd.read_csv(file_path, encoding=encoding)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Validate columns
        self._validate_data(df)

        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate dataframe has required columns

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['Message', 'Category']

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info("Data validation passed")

    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare target variable (convert Category to binary)

        Args:
            df: Input dataframe

        Returns:
            DataFrame with 'is_spam' column
        """
        logger.info("Preparing target variable")

        # Convert 'Category' to binary
        df['is_spam'] = (df['Category'].str.lower() == 'spam').astype(int)

        # Log distribution
        spam_count = df['is_spam'].sum()
        ham_count = len(df) - spam_count
        logger.info(f"Spam messages: {spam_count} ({spam_count/len(df)*100:.2f}%)")
        logger.info(f"Ham messages: {ham_count} ({ham_count/len(df)*100:.2f}%)")

        return df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the dataset

        Args:
            df: Input dataframe

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_messages': len(df),
            'spam_count': df['is_spam'].sum() if 'is_spam' in df.columns else 0,
            'ham_count': len(df) - df['is_spam'].sum() if 'is_spam' in df.columns else 0,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict()
        }

        return summary
