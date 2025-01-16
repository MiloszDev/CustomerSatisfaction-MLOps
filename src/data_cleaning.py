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
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data):
        """
        Preprocess Data
        """
        import logging

class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data):
        """
        Preprocess Data
        """
        try:            
            columns_to_drop = [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ]
            
            columns_to_drop = [col for col in columns_to_drop if col in data.columns]
            
            if columns_to_drop:
                data = data.drop(columns=columns_to_drop, axis=1)
            else:
                logging.warning("No columns to drop. Check if columns exist.")
            
            columns_to_fill = {
                'product_weight_g': data['product_weight_g'].median() if 'product_weight_g' in data.columns else None,
                'product_length_cm': data['product_length_cm'].median() if 'product_length_cm' in data.columns else None,
                'product_height_cm': data['product_height_cm'].median() if 'product_height_cm' in data.columns else None,
                'product_width_cm': data['product_width_cm'].median() if 'product_width_cm' in data.columns else None,
                'review_comment_message': 'No review' if 'review_comment_message' in data.columns else None
            }

            for column, value in columns_to_fill.items():
                if value is not None:
                    data[column] = data[column].fillna(value)

            data = data.select_dtypes(include=[np.number])

            cols_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            data = data.drop(cols_to_drop, axis=1, errors='ignore')

            return data

        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e


class DataSplitStrategy(DataStrategy):
    """
    Strategy for splitting data into train and test sets.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
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
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data using the strategy.
        """
        return self.strategy.handle_data(self.data)

    