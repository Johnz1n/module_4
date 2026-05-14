from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    ticker: str = Field(..., examples=["BBAS3"])
    predicted_close: float = Field(..., description="Preço de fecho previsto em R$", examples=[27.43])
    prediction_date: str = Field(..., description="Próximo dia útil (YYYY-MM-DD)", examples=["2026-05-15"])
    model_version: str = Field(..., examples=["1.0.0"])
    latency_ms: float = Field(..., description="Latência total da requisição em ms")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_predictions: int
    avg_latency_ms: float
    p95_latency_ms: float


class ErrorResponse(BaseModel):
    detail: str
