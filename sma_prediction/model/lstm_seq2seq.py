"""Arquitetura LSTM encoder-decoder (Seq2Seq) para previsão da MA."""

import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder LSTM que resume a janela de contexto em (hidden, cell)."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Codifica a sequência de entrada.

        Args:
            x: tensor ``(batch, lookback, input_size)``.

        Returns:
            Tupla ``(hidden, cell)`` do último passo do encoder.
        """
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    """Decoder LSTM autoregressivo que projeta hidden -> delta da MA."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executa um passo do decoder.

        Args:
            token: tensor ``(batch, 1, 1)`` — entrada do passo atual.
            hidden: estado oculto ``(num_layers, batch, hidden_size)``.
            cell: estado de célula ``(num_layers, batch, hidden_size)``.

        Returns:
            Tupla ``(pred, hidden, cell)`` com ``pred`` shape ``(batch, 1)``.
        """
        out, (hidden, cell) = self.lstm(token, (hidden, cell))
        pred = self.fc(out.squeeze(1))  # (batch, 1)
        return pred, hidden, cell


class LSTMSeq2Seq(nn.Module):
    """Modelo Seq2Seq: encoder resume o contexto, decoder gera H deltas da MA."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        forecast_steps: int,
    ) -> None:
        """Inicializa o modelo.

        Args:
            num_features: número de features de entrada.
            hidden_size: dimensão do estado oculto.
            num_layers: número de camadas LSTM.
            dropout: taxa de dropout entre camadas.
            forecast_steps: horizonte de previsão (nº de passos do decoder).
        """
        super().__init__()
        self.forecast_steps = forecast_steps
        self.encoder = Encoder(num_features, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, num_layers, dropout)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Gera a sequência de ``forecast_steps`` deltas da MA.

        Durante o treino (``target`` fornecido), usa teacher forcing de forma
        estocástica. Na inferência (``target=None``), é totalmente autoregressivo.

        Args:
            x: tensor ``(batch, lookback, num_features)``.
            target: tensor ``(batch, forecast_steps)`` ou ``None`` na inferência.
            teacher_forcing_ratio: probabilidade de usar o valor real como
                próxima entrada do decoder (ignorado se ``target`` é ``None``).

        Returns:
            Tensor ``(batch, forecast_steps)`` com os deltas previstos.
        """
        batch_size = x.size(0)
        hidden, cell = self.encoder(x)

        # Token inicial do decoder: zero (delta de partida)
        token = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(self.forecast_steps):
            pred, hidden, cell = self.decoder(token, hidden, cell)
            outputs.append(pred)

            use_teacher = (
                target is not None and random.random() < teacher_forcing_ratio
            )
            if use_teacher:
                next_val = target[:, t].unsqueeze(1)  # (batch, 1)
            else:
                next_val = pred  # (batch, 1)
            token = next_val.unsqueeze(1)  # (batch, 1, 1)

        return torch.cat(outputs, dim=1)  # (batch, forecast_steps)


def build_model(config: dict, num_features: int) -> LSTMSeq2Seq:
    """Constrói o modelo a partir da configuração.

    Args:
        config: dicionário de configuração.
        num_features: número de features de entrada.

    Returns:
        Instância de ``LSTMSeq2Seq``.
    """
    return LSTMSeq2Seq(
        num_features=num_features,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        forecast_steps=config["forecast_steps"],
    )
