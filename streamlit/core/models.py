"""Arquiteturas de rede neural disponíveis no painel: LSTM, GRU e MLP.

Todas seguem a mesma interface: recebem uma janela ``(batch, lookback,
num_features)`` e produzem, por amostra:

- ``output_size == 1``: um escalar ``(batch,)`` — retorno previsto
  (regressão) ou logit (classificação binária).
- ``output_size > 1``: um vetor de logits ``(batch, output_size)`` — usado
  em classificação multi-classe (ex.: regime de mercado), com
  ``nn.CrossEntropyLoss``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """LSTM many-to-one: usa o último hidden state para prever a saída."""

    def __init__(
        self, num_features: int, hidden_size: int, num_layers: int, dropout: float, output_size: int = 1
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last = hidden[-1]  # (batch, hidden_size)
        out = self.head(last)
        return out.squeeze(-1) if self.output_size == 1 else out


class GRUNet(nn.Module):
    """GRU many-to-one: usa o último hidden state para prever a saída."""

    def __init__(
        self, num_features: int, hidden_size: int, num_layers: int, dropout: float, output_size: int = 1
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        last = hidden[-1]
        out = self.head(last)
        return out.squeeze(-1) if self.output_size == 1 else out


class MLPNet(nn.Module):
    """MLP simples sobre a janela achatada (lookback * num_features)."""

    def __init__(
        self, num_features: int, lookback: int, hidden_size: int, dropout: float, output_size: int = 1
    ) -> None:
        super().__init__()
        self.output_size = output_size
        in_dim = num_features * lookback
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.output_size == 1 else out


def build_model(
    arquitetura: str,
    num_features: int,
    lookback: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    output_size: int = 1,
) -> nn.Module:
    """Constrói o modelo de acordo com a arquitetura escolhida na UI.

    Args:
        arquitetura: um de ``config.ARQUITETURAS`` ("LSTM", "GRU", "MLP").
        num_features: número de colunas de features de entrada.
        lookback: tamanho da janela de contexto (usado só pelo MLP).
        hidden_size: dimensão da camada oculta.
        num_layers: número de camadas recorrentes (LSTM/GRU).
        dropout: taxa de dropout.
        output_size: número de saídas — ``1`` para regressão/classificação
            binária, ``N`` (nº de classes) para classificação multi-classe.

    Returns:
        Instância de ``nn.Module`` pronta para treino.
    """
    if arquitetura == "LSTM":
        return LSTMNet(num_features, hidden_size, num_layers, dropout, output_size)
    if arquitetura == "GRU":
        return GRUNet(num_features, hidden_size, num_layers, dropout, output_size)
    if arquitetura == "MLP":
        return MLPNet(num_features, lookback, hidden_size, dropout, output_size)
    raise ValueError(f"Arquitetura desconhecida: {arquitetura}")
