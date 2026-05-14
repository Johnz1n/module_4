import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.model.lstm import LSTMModel

ARTIFACTS_DIR = Path("artifacts")
EPOCHS = 500
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 100

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_epoch(
    model: LSTMModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _eval_epoch(
    model: LSTMModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item()
    return total_loss / len(loader)


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
) -> LSTMModel:
    """Treina o modelo LSTM e exporta artefactos em artifacts/.

    Guarda model_best.pt (menor val_loss) e model.pt (epoch final).

    Args:
        train_loader: DataLoader do conjunto de treino.
        val_loader: DataLoader do conjunto de validação.
        epochs: Número fixo de epochs de treino.

    Returns:
        Modelo treinado com os pesos da última epoch.
    """
    device = get_device()
    logger.info("Device de treino: %s", device)

    model = LSTMModel().to(device)
    # HuberLoss é mais robusta a outliers que MSELoss: penaliza quadraticamente
    # erros pequenos (<delta) e linearmente erros grandes, reduzindo o impacto
    # de spikes de mercado no gradiente
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=15, factor=0.7, min_lr=1e-4)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(
            "Epoch %03d/%d | train=%.6f | val=%.6f | lr=%.6f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ARTIFACTS_DIR / "model_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping na epoch %d (sem melhoria por %d epochs)", epoch, EARLY_STOP_PATIENCE)
                break

    torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pt")
    logger.info("Treino concluído. Melhor val_loss: %.6f", best_val_loss)
    return model
