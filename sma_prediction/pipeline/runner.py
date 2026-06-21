"""Orquestrador central do pipeline completo de previsão da MA."""

import logging
import os
import time

from data.collector import collect_ohlcv
from data.dataset import build_datasets
from data.features import build_features
from diagnostics.learning_check import diagnose_learning
from diagnostics.plots import generate_backtest_plot, generate_diagnostic_plots
from evaluate.metrics import compute_metrics, gather_predictions, save_metrics
from export.onnx_export import export_to_onnx
from model.lstm_seq2seq import build_model
from model.train import train_model

logger = logging.getLogger(__name__)


def build_run_dir(config: dict) -> str:
    """Monta o caminho do diretório de artefatos baseado nos parâmetros.

    Args:
        config: dicionário de configuração.

    Returns:
        Caminho ``artifacts/{symbol}_{tf}_MA{period}_F{steps}/``.
    """
    name = (
        f"{config['symbol']}_{config['timeframe']}_"
        f"MA{config['ma_period']}_F{config['forecast_steps']}"
    )
    return os.path.join(config["artifact_dir"], name)


def _banner(config: dict, run_dir: str) -> None:
    """Imprime o cabeçalho do pipeline."""
    logger.info("=" * 60)
    logger.info("  MA FORECAST PIPELINE")
    logger.info(
        "  %s | %s | SMA %d | Forecast +%d barras",
        config["symbol"], config["timeframe"], config["ma_period"], config["forecast_steps"],
    )
    logger.info("  Run dir: %s", run_dir)
    logger.info("=" * 60)


def run_pipeline(config: dict) -> dict:
    """Executa todas as etapas do pipeline de treino em sequência.

    Args:
        config: dicionário de configuração.

    Returns:
        Dicionário com ``run_dir``, ``metrics`` e ``diagnostico``.

    Raises:
        Exception: propaga erros de qualquer etapa com contexto no log.
    """
    t_start = time.time()
    run_dir = build_run_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    _banner(config, run_dir)

    # [ETAPA 1/8] Coleta
    logger.info("[ETAPA 1/8] Conectando ao MT5 e coletando dados...")
    df_raw = collect_ohlcv(config, run_dir)

    # [ETAPA 2/8] Features
    logger.info("[ETAPA 2/8] Calculando features e salvando dataset...")
    df_feat = build_features(df_raw, config)
    feat_path = os.path.join(run_dir, "dataset_features.csv")
    df_feat.to_csv(feat_path, index=False)
    logger.info("💾 Salvo: %s", feat_path)

    # [ETAPA 3/8] Janelas e splits
    logger.info("[ETAPA 3/8] Criando janelas deslizantes e splits...")
    data = build_datasets(df_feat, config, run_dir)

    # [ETAPA 4/8] Treino
    logger.info("[ETAPA 4/8] Treinando modelo...")
    model = build_model(config, data["num_features"])
    train_result = train_model(
        model, data["train_loader"], data["val_loader"], config, run_dir
    )

    # [ETAPA 5/8] Avaliação
    logger.info("[ETAPA 5/8] Avaliando no conjunto de teste...")
    preds = gather_predictions(
        model, data["test_ds"], train_result["device"], config["batch_size"]
    )
    metrics = compute_metrics(
        preds["y_true_price"], preds["y_pred_price"], preds["anchor_ma"], config
    )
    save_metrics(metrics, run_dir)
    g = metrics["global"]
    logger.info("  MAE global     : %s pips", g["mae_pips"])
    logger.info("  RMSE global    : %s pips", g["rmse_pips"])
    logger.info("  Dir. Accuracy  : %.1f%%", g["directional_acc"] * 100)

    # [ETAPA 6/8] Export ONNX
    logger.info("[ETAPA 6/8] Exportando modelo para ONNX...")
    export_to_onnx(model, config, run_dir, data["scaler"], data["num_features"])

    # [ETAPA 7/8] Gráficos de diagnóstico
    logger.info("[ETAPA 7/8] Gerando gráficos de diagnóstico...")
    splits = {"train_end": data["train_end"], "val_end": data["val_end"]}
    generate_diagnostic_plots(
        df_feat, splits, train_result["loss_history"], preds, metrics, config, run_dir
    )

    # [ETAPA 8/8] Backtest overlay
    logger.info("[ETAPA 8/8] Gerando gráfico de comparação com preço real...")
    generate_backtest_plot(preds, metrics, config, run_dir)

    diag = diagnose_learning(train_result["loss_history"])
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)

    logger.info("=" * 60)
    logger.info("  ✅ PIPELINE CONCLUÍDO")
    logger.info("  Tempo total: %dm %ds", mins, secs)
    logger.info("  Artefatos: %s", run_dir)
    logger.info("  Diagnóstico: %s %s", diag["emoji"], diag["titulo"])
    logger.info("=" * 60)

    return {"run_dir": run_dir, "metrics": metrics, "diagnostico": diag}
