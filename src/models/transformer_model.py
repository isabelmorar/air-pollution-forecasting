import math

import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):
    """Single-step PM2.5 forecaster using a Transformer encoder with sinusoidal positional encoding."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
        self.fc = nn.Linear(d_model, 1)
        self.d_model = d_model

    def _positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Add fixed sinusoidal positional encoding to the input embeddings."""
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)       # (seq_len, 1)
        dims = torch.arange(0, self.d_model, 2, device=x.device).float()      # (d_model/2,)
        div_term = torch.exp(dims * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return x + pe.unsqueeze(0)  # broadcast over batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer encoder and final prediction layer."""
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)           # projects to shape (batch, seq_len, d_model)
        x = self._positional_encoding(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)  # last token → (batch,)
