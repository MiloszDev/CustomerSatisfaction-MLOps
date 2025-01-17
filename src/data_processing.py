import logging
import numpy as np
import pandas as pd

from typing import Union
from abc import ABC, abstractmethod
from sklearn.impute import SimpleImputer
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
            data['deliver_time'] = (pd.to_datetime(data['order_delivered_customer_date']) - pd.to_datetime(data['order_purchase_timestamp'])).dt.days
            data['product_volume'] = data['product_length_cm'] * data['product_height_cm'] * data['product_width_cm']

            data = self.drop_columns(data)
            data = self.fill_missing_values(data)
            return self.select_numerical_data(data)
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise e
    
    def drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        data.drop(columns=['order_id', 'customer_id', 'order_item_id', 'order_approved_at', 'order_delivered_carrier_date', 'order_estimated_delivery_date', 'review_comment_message'], inplace=True, errors='ignore')
        return data

    def fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_fill = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
        
        for col in cols_to_fill:
            data[col] = data[col].fillna(data[col].median()) 
        
        if 'order_delivered_carrier_date' in data.columns:
            data['order_delivered_carrier_date'] = data['order_delivered_carrier_date'].fillna(pd.NaT)
        
        if 'order_delivered_customer_date' in data.columns:
            data['order_delivered_customer_date'] = data['order_delivered_customer_date'].fillna(pd.NaT)
        
        return data

    def select_numerical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.select_dtypes(include=[np.number])

class DataSplitStrategy(DataStrategy):
    """
    Strategy for splitting data into train and test sets.
    """

    def process_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            imputer = SimpleImputer(strategy='mean')
            
            X = pd.DataFrame(imputer.fit_transform(data.drop('review_score', axis=1)))
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
