import logging
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    """
    Abstract class for all models.
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model.

        Args:
            X_train: the training data
            y_train: the training labels
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Model:
        """
        Trains the Linear Regression model.

        Args:
            X_train: the training data
            y_train: the training labels
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            
            logging.info("Training Linear Regression model.")

            return reg

        except Exception as e:
            logging.error(f"Error in training Linear Regression model: {e}")
            raise e
        
class RandomForestRegressorModel(Model):
    """
    Random Forest Regressor Model.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Model:
        """
        Trains the Random Forest Regressor model.

        Args:
            X_train: the training data
            y_train: the training labels
        Returns:
            None
        """
        try:
            reg = RandomForestRegressor(**kwargs)
            reg.fit(X_train, y_train)
            
            logging.info("Training Random Forest Regressor model.")

            return reg

        except Exception as e:
            logging.error(f"Error in training Random Forest Regressor model: {e}")
            raise e