import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

BBAS3_TICKER = "BBAS3.SA"
IBOV_TICKER = "^BVSP"
DEFAULT_PERIOD = "5y"
RAW_DATA_DIR = Path("data/raw")

logger = logging.getLogger(__name__)


def _download_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    # yfinance >= 0.2.x retorna MultiIndex ao baixar múltiplos tickers de uma vez;
    # ao baixar individualmente pode variar conforme a versão — achatamos para garantir
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def fetch_and_align(period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Baixa BBAS3 e IBOV e retorna DataFrame alinhado por data (inner join).

    Args:
        period: Período aceito pelo yfinance (ex: '5y', '2y').

    Returns:
        DataFrame com colunas OHLCV do BBAS3 + IBOV_Close, sem NaN.
    """
    bbas3 = _download_ohlcv(BBAS3_TICKER, period)
    ibov = _download_ohlcv(IBOV_TICKER, period)[["Close"]].rename(
        columns={"Close": "IBOV_Close"}
    )

    merged = bbas3.join(ibov, how="inner")
    merged.dropna(inplace=True)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    bbas3.to_csv(RAW_DATA_DIR / "BBAS3.csv")
    ibov.to_csv(RAW_DATA_DIR / "IBOV.csv")

    logger.info("Dias de pregão disponíveis: %d", len(merged))
    return merged
