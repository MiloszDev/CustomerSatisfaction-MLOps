import logging
import pandas as pd

from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.data_processing import DataCleaner, DataPreprocessStrategy, DataSplitStrategy

@step
def clean_data(df: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, 'X_train'], 
                                          Annotated[pd.DataFrame, 'X_test'], 
                                          Annotated[pd.Series, 'y_train'], 
                                          Annotated[pd.Series, 'y_test']]:
    """
    Cleans the data and splits it into train and test sets.
    
    Args:
        df: the ingested data
    Returns:
        Tuple containing X_train, X_test, y_train, and y_test
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaner = DataCleaner(df, preprocess_strategy)
        
        processed_data = data_cleaner.process_data()

        split_strategy = DataSplitStrategy()
        data_cleaner = DataCleaner(processed_data, split_strategy)

        logging.info("Data cleaning and preprocessing complete.")
        
        X_train, X_test, y_train, y_test = data_cleaner.process_data()

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e