import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from src.data.dataset import LOOKBACK, inverse_transform_close
from src.data.features import FEATURE_COLUMNS, N_FEATURES, build_features
from src.model.lstm import LSTMModel

ARTIFACTS_DIR = Path("artifacts")
MODEL_VERSION = "1.0.0"
# 60 dias garante ao menos 30 válidos após o cálculo do MACD(26)
FETCH_PERIOD = "60d"

logger = logging.getLogger(__name__)


class Predictor:
    """Encapsula o modelo treinado e a lógica de inferência em produção."""

    def __init__(self) -> None:
        self.scaler: MinMaxScaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
        self.model = LSTMModel()
        state = torch.load(
            ARTIFACTS_DIR / "model_best.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, ticker: str) -> dict:
        """Executa o pipeline completo de inferência para o ticker informado.

        Args:
            ticker: Código do ativo sem sufixo (ex: 'BBAS3').

        Returns:
            Dicionário com previsão e metadados da requisição.

        Raises:
            ValueError: Se os dados baixados forem insuficientes para a janela.
        """
        start = datetime.now()

        window = self._build_input_window(ticker)
        normalized = self.scaler.transform(window)

        input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            raw_output = self.model(input_tensor).item()

        predicted_price = inverse_transform_close(self.scaler, np.array([raw_output]))[0]
        latency_ms = (datetime.now() - start).total_seconds() * 1000

        return {
            "ticker": ticker.upper(),
            "predicted_close": round(float(predicted_price), 2),
            "prediction_date": _next_business_day(),
            "model_version": MODEL_VERSION,
            "latency_ms": round(latency_ms, 2),
        }

    def _build_input_window(self, ticker: str) -> np.ndarray:
        """Baixa dados recentes, calcula features e retorna a janela de 30 dias."""
        bbas3_raw = _download_ohlcv(f"{ticker}.SA", FETCH_PERIOD)
        ibov_raw = _download_ohlcv("^BVSP", FETCH_PERIOD)

        ibov = ibov_raw[["Close"]].rename(columns={"Close": "IBOV_Close"})
        merged = bbas3_raw.join(ibov, how="inner").dropna()

        features_df = build_features(merged)

        if len(features_df) < LOOKBACK:
            raise ValueError(
                f"Dados insuficientes para '{ticker}': {len(features_df)} dias "
                f"disponíveis após cálculo de indicadores (mínimo: {LOOKBACK})."
            )

        return features_df[FEATURE_COLUMNS].values[-LOOKBACK:].astype(np.float32)


def _download_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _next_business_day() -> str:
    today = datetime.now()
    # Pula para segunda se hoje for sexta (4) ou sábado (5)
    days_ahead = 1 if today.weekday() < 4 else (7 - today.weekday())
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
