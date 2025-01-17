import logging
import numpy as np
import pandas as pd

from typing import Union
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data and cleaning it.
    """
    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    """Strategy to preprocess the data."""
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        try:
            data = self.drop_columns(data)
            data = self.fill_missing_values(data)
            return self.select_numerical_data(data)
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise e
    
    def drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = ["order_approved_at", "order_delivered_carrier_date", ...]
        return data.drop(columns=columns_to_drop, axis=1, errors='ignore')

    def fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        columns_to_fill = {
            'product_weight_g': data['product_weight_g'].median(),
            'review_comment_message': 'No review'
        }
        for column, value in columns_to_fill.items():
            data[column] = data[column].fillna(value)
        return data

    def select_numerical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.select_dtypes(include=[np.number])

class DataSplitStrategy(DataStrategy):
    """
    Strategy for splitting data into train and test sets.
    """

    def process_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop('review_score', axis=1)
            y = data['review_score']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in splitting the data: {e}")
            raise e

class DataCleaner:
    """
    Class to clean and preprocess data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def process_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data using the strategy.
        """
        return self.strategy.process_data(self.data)