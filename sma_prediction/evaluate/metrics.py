"""Métricas de avaliação calculadas sobre valores desnormalizados da MA."""

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import SlidingWindowDataset
from model.lstm_seq2seq import LSTMSeq2Seq

logger = logging.getLogger(__name__)


def _pip_size(symbol: str) -> float:
    """Estima o tamanho de 1 pip para o símbolo.

    Args:
        symbol: nome do ativo.

    Returns:
        Tamanho de 1 pip em unidades de preço.
    """
    sym = symbol.upper()
    if "JPY" in sym:
        return 0.01
    if sym.startswith("XAU") or "GOLD" in sym:
        return 0.1
    return 0.0001


def gather_predictions(
    model: LSTMSeq2Seq,
    dataset: SlidingWindowDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Roda o modelo (inferência, sem teacher forcing) sobre um dataset.

    Converte os deltas previstos de volta para preço da MA usando a âncora:
    ``ma_prevista[h] = ma_ancora + y_pred[h] * atr_ancora``.

    Args:
        model: modelo treinado.
        dataset: ``SlidingWindowDataset`` (geralmente o de teste).
        device: device de execução.
        batch_size: tamanho do batch.

    Returns:
        Dicionário com ``y_true_price``, ``y_pred_price`` (shape ``(N, H)``),
        ``anchor_ma``, ``anchor_atr``, ``anchor_time`` e ``y_true_delta``.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x, target=None, teacher_forcing_ratio=0.0)
            preds.append(out.cpu().numpy())

    y_pred_delta = np.concatenate(preds, axis=0) if preds else np.empty((0, 0))

    anchors = dataset.anchor_arrays()
    anchor_ma = anchors["ma"].astype(np.float64)
    anchor_atr = anchors["atr"].astype(np.float64)
    y_true_delta = anchors["targets"].astype(np.float64)

    # Desnormalização: delta/ATR -> preço da MA
    y_pred_price = anchor_ma[:, None] + y_pred_delta * anchor_atr[:, None]
    y_true_price = anchor_ma[:, None] + y_true_delta * anchor_atr[:, None]

    return {
        "y_true_price": y_true_price,
        "y_pred_price": y_pred_price,
        "y_true_delta": y_true_delta,
        "y_pred_delta": y_pred_delta,
        "anchor_ma": anchor_ma,
        "anchor_atr": anchor_atr,
        "anchor_time": anchors["time"],
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anchor_ma: np.ndarray,
    config: dict,
) -> dict:
    """Calcula métricas por horizonte e globais sobre preços desnormalizados.

    Args:
        y_true: preços reais da MA, shape ``(N, H)``.
        y_pred: preços previstos da MA, shape ``(N, H)``.
        anchor_ma: MA na barra-âncora, shape ``(N,)`` (baseline direcional).
        config: dicionário de configuração (symbol, timeframe, etc.).

    Returns:
        Dicionário de métricas no formato do ``metrics_test.json``.
    """
    pip = _pip_size(config["symbol"])
    horizon = y_true.shape[1] if y_true.ndim == 2 else 0

    by_horizon = []
    for h in range(horizon):
        err = y_pred[:, h] - y_true[:, h]
        mae = np.mean(np.abs(err)) / pip
        rmse = np.sqrt(np.mean(err**2)) / pip

        true_dir = np.sign(y_true[:, h] - anchor_ma)
        pred_dir = np.sign(y_pred[:, h] - anchor_ma)
        dir_acc = float(np.mean(true_dir == pred_dir)) if len(true_dir) else 0.0

        by_horizon.append(
            {
                "h": h + 1,
                "mae_pips": round(float(mae), 2),
                "rmse_pips": round(float(rmse), 2),
                "directional_acc": round(dir_acc, 4),
            }
        )

    global_metrics = {
        "mae_pips": round(float(np.mean([b["mae_pips"] for b in by_horizon])), 2),
        "rmse_pips": round(float(np.mean([b["rmse_pips"] for b in by_horizon])), 2),
        "directional_acc": round(
            float(np.mean([b["directional_acc"] for b in by_horizon])), 4
        ),
    }

    return {
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "ma_period": config["ma_period"],
        "forecast_steps": config["forecast_steps"],
        "by_horizon": by_horizon,
        "global": global_metrics,
    }


def save_metrics(metrics: dict, run_dir: str) -> str:
    """Salva as métricas em ``run_dir/metrics_test.json``.

    Args:
        metrics: dicionário de métricas.
        run_dir: diretório de artefatos.

    Returns:
        Caminho do arquivo salvo.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "metrics_test.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path
