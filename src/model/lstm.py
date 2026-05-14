import torch
import torch.nn as nn

from src.data.features import N_FEATURES


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :])
