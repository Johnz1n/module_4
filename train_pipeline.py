"""Ponto de entrada para o pipeline completo de treino."""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
    force=True,
)

from src.data.dataset import build_dataloaders
from src.data.fetch import fetch_and_align
from src.data.features import build_features
from src.evaluate import evaluate
from src.model.train import train


def main() -> None:
    logging.info("=== Pipeline de treino BBAS3-LSTM ===")

    raw_df = fetch_and_align()
    features_df = build_features(raw_df)

    logging.info(
        "Features prontas: %d amostras | %s → %s",
        len(features_df),
        features_df.index[0].date(),
        features_df.index[-1].date(),
    )

    train_loader, val_loader, test_loader, scaler = build_dataloaders(features_df)
    model = train(train_loader, val_loader)

    metrics = evaluate(model, test_loader, scaler)

    logging.info("=== Resultados no Test Set ===")
    logging.info("MAE:  R$ %.4f", metrics["MAE"])
    logging.info("RMSE: R$ %.4f", metrics["RMSE"])
    logging.info("MAPE: %.2f%%", metrics["MAPE"])


if __name__ == "__main__":
    main()
