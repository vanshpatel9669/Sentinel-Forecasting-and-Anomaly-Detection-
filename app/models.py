from pydantic import BaseModel

class SeriesRequest(BaseModel):
    metric_name: str
    values: list[float]
    forecast_horizon: int = 5

class AnalysisResponse(BaseModel):
    metric_name: str
    forecast: list[float]
    anomaly_indices: list[int]
    alert_level: str
    summary: str
