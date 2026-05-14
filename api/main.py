import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import Predictor
from api.schemas import ErrorResponse, HealthResponse, MetricsResponse, PredictionResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

_predictor: Predictor | None = None
_start_time: float = 0.0
_prediction_count: int = 0
_latency_window: deque[float] = deque(maxlen=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _start_time
    _start_time = time.time()
    _predictor = Predictor()
    logger.info('{"event": "startup", "status": "modelo carregado com sucesso"}')
    yield
    logger.info('{"event": "shutdown"}')


app = FastAPI(
    title="BBAS3 LSTM Predictor",
    description=(
        "API de previsão de preços para ativos da B3 utilizando redes LSTM.\n\n"
        "O modelo foi treinado com dados históricos OHLCV + indicadores técnicos "
        "(RSI, MACD, Bollinger Bands, EMA) e dado exógeno (Ibovespa)."
    ),
    version=os.getenv("MODEL_VERSION", "1.0.0"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Dados insuficientes ou ticker inválido"},
        500: {"model": ErrorResponse, "description": "Erro interno de inferência"},
    },
    summary="Prever preço de fecho",
    tags=["Predição"],
)
async def predict(
    ticker: Annotated[
        str,
        Query(description="Código do ativo B3 sem sufixo .SA", example="BBAS3"),
    ] = "BBAS3",
) -> PredictionResponse:
    """Prevê o preço de fecho do próximo dia útil para o ticker informado.

    Faz o download dos últimos 60 dias via yfinance, calcula os indicadores
    técnicos e executa a inferência com o modelo LSTM treinado.
    """
    global _prediction_count

    try:
        result = _predictor.predict(ticker)
        _prediction_count += 1
        _latency_window.append(result["latency_ms"])
        logger.info(
            '{"event": "prediction", "ticker": "%s", "predicted_close": %s, "latency_ms": %s}',
            result["ticker"],
            result["predicted_close"],
            result["latency_ms"],
        )
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error('{"event": "error", "detail": "%s"}', str(exc))
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição.") from exc


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado da API",
    tags=["Operações"],
)
async def health() -> HealthResponse:
    """Verifica se a API está operacional e o modelo está carregado."""
    return HealthResponse(
        status="ok" if _predictor is not None else "degraded",
        model_loaded=_predictor is not None,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Métricas de uso",
    tags=["Operações"],
)
async def metrics() -> MetricsResponse:
    """Retorna contadores de predições e estatísticas de latência."""
    latencies = list(_latency_window)
    return MetricsResponse(
        total_predictions=_prediction_count,
        avg_latency_ms=round(float(np.mean(latencies)), 2) if latencies else 0.0,
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    )
