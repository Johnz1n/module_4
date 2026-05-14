import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import inverse_transform_close
from src.model.lstm import LSTMModel

ARTIFACTS_DIR = Path("artifacts")

logger = logging.getLogger(__name__)


def _collect_predictions(
    model: LSTMModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.inference_mode():
        for x, y in loader:
            preds.extend(model(x.to(device)).cpu().numpy().flatten())
            targets.extend(y.numpy().flatten())
    return np.array(preds), np.array(targets)


def _compute_metrics(targets: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(targets - preds)))
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    mape = float(np.mean(np.abs((targets - preds) / targets)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _save_plot(targets: np.ndarray, preds: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(targets, label="Real", alpha=0.85)
    ax.plot(preds, label="Previsto", alpha=0.85, linestyle="--")
    ax.set_title("BBAS3 — Preço Real vs. Previsto (Test Set)")
    ax.set_ylabel("Preço (R$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ARTIFACTS_DIR / "evaluation_plot.png", dpi=150)
    plt.close(fig)
    logger.info("Gráfico salvo em %s", ARTIFACTS_DIR / "evaluation_plot.png")


def evaluate(
    model: LSTMModel,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
) -> dict[str, float]:
    """Avalia o modelo no test set e gera gráfico de comparação.

    Args:
        model: Modelo LSTM treinado.
        test_loader: DataLoader do conjunto de teste.
        scaler: MinMaxScaler fitado no treino (para inverse_transform).

    Returns:
        Dicionário com MAE (R$), RMSE (R$) e MAPE (%).
    """
    device = next(model.parameters()).device
    norm_preds, norm_targets = _collect_predictions(model, test_loader, device)

    preds = inverse_transform_close(scaler, norm_preds)
    targets = inverse_transform_close(scaler, norm_targets)

    metrics = _compute_metrics(targets, preds)
    _save_plot(targets, preds)

    return metrics
