import logging
import pandas as pd

from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data file
        """
        self.data_path = data_path
    
    def get_data(self):
        """
        Ingesting the data from the data_path.
        """
        logging.info(f'Ingesting data from: {self.data_path}')
        return pd.read_csv(self.data_path, index_col=0, parse_dates=True, data_parser=lambda x:pd.to_datetime(x, format='%Y-%m-%d'))

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path.

    Args:
        data_path: path to the data file
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        return IngestData(data_path).get_data()
    
    except Exception as e:
        logging.error(f'Error ingesting data: {e}')
        return e