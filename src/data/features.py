import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "Open",         # Preço de abertura do pregão
    "High",         # Máxima intraday
    "Low",          # Mínima intraday
    "Close",        # Preço de fecho — target do modelo
    "Volume",       # Volume financeiro negociado
    "RSI_14",       # Índice de Força Relativa (14 períodos): mede momentum; >70 sobrecomprado, <30 sobrevendido
    "MACD_Line",    # Diferença entre EMA(12) e EMA(26): indica direção da tendência
    "MACD_Signal",  # EMA(9) da MACD Line: sinal de compra/venda no cruzamento
    "MACD_Hist",    # MACD Line − Signal: histograma da força da tendência
    "BB_Width",     # Largura das Bandas de Bollinger: mede expansão/contração da volatilidade
    "EMA_9",        # Média Móvel Exponencial de 9 dias: tendência de curto prazo
    "EMA_21",       # Média Móvel Exponencial de 21 dias: tendência de médio prazo
    "Volume_Ratio", # Volume atual / SMA(20) do volume: detecta anomalias de liquidez
    "IBOV_Close",   # Fecho do Ibovespa (^BVSP): proxy do humor geral do mercado brasileiro
    "Close_pct",    # Variação percentual diária do Close: sinal direcional explícito para o LSTM
]

CLOSE_IDX: int = FEATURE_COLUMNS.index("Close")  # 3
N_FEATURES: int = len(FEATURE_COLUMNS)           # 15


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger_width(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ((mid + std_dev * std) - (mid - std_dev * std)) / mid


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume / volume.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula os 15 indicadores técnicos e retorna DataFrame limpo sem NaN.

    Args:
        df: DataFrame com colunas OHLCV + IBOV_Close (saída de fetch_and_align).

    Returns:
        DataFrame com FEATURE_COLUMNS na ordem definida, sem NaN.
    """
    result = df.copy()

    result["RSI_14"] = _rsi(result["Close"])
    result["MACD_Line"], result["MACD_Signal"], result["MACD_Hist"] = _macd(result["Close"])
    result["BB_Width"] = _bollinger_width(result["Close"])
    result["EMA_9"] = _ema(result["Close"], 9)
    result["EMA_21"] = _ema(result["Close"], 21)
    result["Volume_Ratio"] = _volume_ratio(result["Volume"])
    result["Close_pct"] = result["Close"].pct_change()

    return result[FEATURE_COLUMNS].dropna()
