from pydantic import BaseModel, ConfigDict

class ModelNameConfig(BaseModel):
    """Model Configs"""
    model_name: str = "RandomForestRegressor"

    model_config = ConfigDict(protected_namespaces=())