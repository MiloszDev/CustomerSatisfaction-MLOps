import logging
import pandas as pd

from zenml import step
from .config import ModelNameConfig
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel

@step
def train_model(X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame,
                config: ModelNameConfig) -> RegressorMixin: # mixin class for all regression estimators in scikit-learn
    """
    Trains the model on the ingested data.

    Args:
        X_train: the training data
        X_test: the test data
        y_train: the training labels
        y_test: the test labels
    """
    model = None

    try:
        if config.model_name == 'LinearRegression':
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)

            return trained_model
        
        else:
            raise ValueError(f"Model {config.model_name} not supported.")
    
    except Exception as e:
        logging.error(f"Error in training model.")
        raise e
