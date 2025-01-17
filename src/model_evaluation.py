import logging
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class for defining strategy for evaluating models.
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            dict: Dictionary containing scores
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Mean Squared Error.")
            
            mse = mean_squared_error(y_true, y_pred)

            logging.info(f"Mean Squared Error: {mse}")
            return mse
        
        except Exception as e:
            logging.error(f"Error in calculating Mean Squared Error: {e}")
            raise e

class R2Score(Evaluation):
    """
    Evaluation strategy that uses R2 Score.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating R2 Score.")
            
            r2 = r2_score(y_true, y_pred)

            logging.info(f"R2 Score: {r2}")
            return r2
        
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Root Mean Squared Error.")

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        
        except Exception as e:
            logging.error(f"Error in calculating Root Mean Squared Error: {e}")
            raise e