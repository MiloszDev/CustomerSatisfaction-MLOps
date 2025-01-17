import logging
import pandas as pd

from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from src.model_evaluation import MSE, RMSE, R2Score

@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[Annotated[float, 'r2'],
                                                  Annotated[float, 'rmse']]:
    """
    Evaluates the model on the ingested data.

    Args:
        df: the ingested data
    """
    try:
        y_pred = model.predict(X_test)

        mse = MSE().calculate_scores(y_test, y_pred)
        r2 = R2Score().calculate_scores(y_test, y_pred)
        rmse = RMSE().calculate_scores(y_test, y_pred)

        return r2, rmse
    
    except Exception as e:
        logging.error(f"Error in evaluating model.")
        raise e