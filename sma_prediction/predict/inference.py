"""Inferência com modelo treinado: previsão das próximas barras da MA.

Usa o modelo exportado em ONNX (mais rápido e valida o export) e replica o
pré-processamento do treino com o scaler salvo.
"""

import json
import logging
import os
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.collector import collect_ohlcv
from data.features import build_features

logger = logging.getLogger(__name__)


def _load_artifacts(run_dir: str) -> tuple[dict, object]:
    """Carrega metadados ONNX e scaler do run_dir.

    Args:
        run_dir: diretório do run treinado.

    Returns:
        Tupla ``(metadata, scaler)``.

    Raises:
        FileNotFoundError: se faltar algum artefato necessário.
    """
    required = ["model.onnx", "onnx_metadata.json", "scaler.pkl"]
    faltando = [f for f in required if not os.path.exists(os.path.join(run_dir, f))]
    if faltando:
        raise FileNotFoundError(
            f"Artefatos ausentes em {run_dir}: {faltando}. Treine primeiro com "
            "--mode train."
        )
    with open(os.path.join(run_dir, "onnx_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    scaler = joblib.load(os.path.join(run_dir, "scaler.pkl"))
    return metadata, scaler


def predict_next(run_dir: str, config: dict) -> dict:
    """Coleta dados recentes do MT5 e prevê as próximas barras da MA.

    Args:
        run_dir: diretório do run treinado (com model.onnx, scaler, metadados).
        config: dicionário de configuração (usado para coleta).

    Returns:
        Dicionário com a previsão formatada (ver docstring do módulo/CLAUDE.md).
    """
    import onnxruntime as ort

    metadata, scaler = _load_artifacts(run_dir)
    lookback = metadata["lookback_window"]
    feature_order = metadata["feature_order"]
    horizon = metadata["forecast_steps"]

    # Coleta recente: barras suficientes para janelas + cálculo de features
    df_raw = collect_ohlcv(config, run_dir)
    df_feat = build_features(df_raw, config)
    if len(df_feat) < lookback:
        raise ValueError(
            f"Dados insuficientes para inferência: {len(df_feat)} < {lookback}."
        )

    window = df_feat.iloc[-lookback:]
    x = scaler.transform(window[feature_order].values).astype(np.float32)
    x = x[None, :, :]  # (1, lookback, num_features)

    session = ort.InferenceSession(
        os.path.join(run_dir, "model.onnx"), providers=["CPUExecutionProvider"]
    )
    y_pred = session.run(["ma_delta"], {"features": x})[0][0]  # (horizon,)

    anchor = df_feat.iloc[-1]
    current_ma = float(anchor["ma"])
    atr = float(anchor["atr"])

    forecast = []
    for h in range(horizon):
        ma_val = current_ma + float(y_pred[h]) * atr
        forecast.append({"bar": h + 1, "ma_value": round(ma_val, 5)})

    direction = "UP" if forecast[-1]["ma_value"] >= current_ma else "DOWN"
    result = {
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "ma_period": config["ma_period"],
        "forecast_steps": horizon,
        "current_ma": round(current_ma, 5),
        "forecast": forecast,
        "direction": direction,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    _plot_forecast(df_feat, result, run_dir, config)
    return result


def _plot_forecast(df_feat: pd.DataFrame, result: dict, run_dir: str, config: dict) -> str:
    """Plota a MA recente e a previsão das próximas barras.

    Args:
        df_feat: DataFrame com features e coluna ``ma``.
        result: dicionário de resultado de ``predict_next``.
        run_dir: diretório de artefatos.
        config: dicionário de configuração.

    Returns:
        Caminho do PNG salvo.
    """
    try:
        plt.style.use(config.get("plot_style", "seaborn-v0_8-darkgrid"))
    except OSError:
        plt.style.use("ggplot")

    tail = df_feat.iloc[-100:]
    t = pd.to_datetime(tail["time"])
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(t, tail["ma"], color="tab:blue", lw=1.4, label="MA real (recente)")

    step = t.iloc[-1] - t.iloc[-2] if len(t) > 1 else pd.Timedelta(hours=1)
    fut_t = [t.iloc[-1] + step * (i + 1) for i in range(len(result["forecast"]))]
    fut_v = [f["ma_value"] for f in result["forecast"]]
    ax.plot(
        [t.iloc[-1]] + fut_t, [result["current_ma"]] + fut_v,
        color="tab:orange", ls="--", marker="o", lw=1.4, label="Previsão NN",
    )
    ax.set_title(
        f"Forecast em tempo real — {config['symbol']} {config['timeframe']} "
        f"SMA{config['ma_period']} (+{result['forecast_steps']}) | {result['direction']}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "09_live_forecast.png")
    fig.savefig(path, dpi=config.get("plot_dpi", 150), bbox_inches="tight")
    plt.close(fig)
    logger.info("✅ 09_live_forecast.png")
    return path
