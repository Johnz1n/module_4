from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from src.data.features import CLOSE_IDX, N_FEATURES

LOOKBACK: int = 30
ARTIFACTS_DIR = Path("artifacts")


class StockDataset(Dataset):
    def __init__(self, scaled_data: np.ndarray, lookback: int = LOOKBACK) -> None:
        self.data = scaled_data
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.data) - self.lookback

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback, CLOSE_IDX]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )


def build_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler]:
    """Cria DataLoaders com split cronológico e normalização sem data leakage.

    O scaler é fitado exclusivamente no conjunto de treino e serializado em
    artifacts/scaler.pkl para uso posterior na API.

    Args:
        df: DataFrame com FEATURE_COLUMNS (saída de build_features).
        batch_size: Tamanho do batch para os DataLoaders.
        train_ratio: Proporção dos dados para treino.
        val_ratio: Proporção dos dados para validação.

    Returns:
        Tupla (train_loader, val_loader, test_loader, scaler).
    """
    values = df.values.astype(np.float32)
    n = len(values)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(values[:train_end])
    val_scaled = scaler.transform(values[train_end:val_end])
    test_scaled = scaler.transform(values[val_end:])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")

    train_loader = DataLoader(StockDataset(train_scaled), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StockDataset(val_scaled), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(StockDataset(test_scaled), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def inverse_transform_close(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """Converte valores normalizados do Close para preços reais em R$.

    Reconstrói o array dummy de 14 features, coloca o valor na posição
    correta e aplica inverse_transform, extraindo apenas a coluna Close.
    """
    dummy = np.zeros((len(values), N_FEATURES), dtype=np.float32)
    dummy[:, CLOSE_IDX] = values
    return scaler.inverse_transform(dummy)[:, CLOSE_IDX]
