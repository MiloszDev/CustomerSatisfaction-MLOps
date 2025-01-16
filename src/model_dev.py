import logging

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
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
    def train(self, X_train, y_train, **kwargs):
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
        
        except Exception as e:
            logging.error(f"Error in training Linear Regression model: {e}")
            raise e