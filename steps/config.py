from pydantic import BaseModel
from zenml.steps import step

class ModelNameConfig(BaseModel):
    """Model Configs"""
    model_name: str = "LinearRegression"