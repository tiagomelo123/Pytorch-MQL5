"""Orquestrador central do pipeline completo de previsão da MA."""

import json
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


def _model_exists(run_dir: str) -> bool:
    """Verifica se os três artefatos obrigatórios já existem no run_dir.

    Args:
        run_dir: diretório de artefatos do run.

    Returns:
        True se model.onnx, onnx_metadata.json e scaler.pkl estiverem presentes.
    """
    required = ["model.onnx", "onnx_metadata.json", "scaler.pkl"]
    return all(os.path.exists(os.path.join(run_dir, f)) for f in required)


def _prompt_existing_model(run_dir: str, config: dict) -> int:
    """Exibe informações do modelo salvo e pede ao usuário o que fazer.

    Args:
        run_dir: diretório de artefatos do run.
        config: dicionário de configuração.

    Returns:
        1 = usar modelo existente, 2 = retreinar, 3 = cancelar.
    """
    onnx_path = os.path.join(run_dir, "model.onnx")
    trained_at = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(onnx_path))
    )
    onnx_kb = os.path.getsize(onnx_path) // 1024

    mae_str = dir_str = "N/A"
    metrics_path = os.path.join(run_dir, "metrics_test.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        g = m.get("global", {})
        mae_str = f"{g.get('mae_pips', 'N/A')} pips"
        dir_acc = g.get("directional_acc", None)
        dir_str = f"{dir_acc * 100:.1f}%" if dir_acc is not None else "N/A"

    print("\n" + "=" * 60)
    print(
        f"  {config['symbol']} | {config['timeframe']} | "
        f"SMA {config['ma_period']} | Forecast +{config['forecast_steps']} barras"
    )
    print("=" * 60)
    print(f"\n⚠️  Modelo já treinado encontrado em: {run_dir}/")
    print(f"   Treinado em : {trained_at}")
    print(f"   MAE test    : {mae_str}")
    print(f"   Dir. Acc    : {dir_str}")
    print(f"   ONNX        : model.onnx ({onnx_kb} KB)")
    print(
        "\n   O que deseja fazer?\n"
        "   [1] Usar modelo existente → gerar apenas inferência e gráficos\n"
        "   [2] Treinar novamente     → substituir modelo atual\n"
        "   [3] Cancelar\n"
    )
    while True:
        choice = input("Escolha (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            return int(choice)
        print("   Opção inválida. Digite 1, 2 ou 3.")


def _run_inference_only(config: dict, run_dir: str, t_start: float) -> dict:
    """Executa apenas a inferência com o modelo já treinado (opção 1 do menu).

    Args:
        config: dicionário de configuração.
        run_dir: diretório de artefatos.
        t_start: timestamp de início (para medir tempo total).

    Returns:
        Dicionário com ``run_dir`` e ``result`` da inferência.
    """
    from predict.inference import predict_next

    logger.info("✅ Modelo existente carregado")
    logger.info("⏭  Pulando etapas 1 a 8 — executando apenas inferência ao vivo\n")

    result = predict_next(run_dir, config)

    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    logger.info("=" * 60)
    logger.info("  ✅ INFERÊNCIA CONCLUÍDA")
    logger.info("  Tempo total: %dm %ds", mins, secs)
    logger.info("  Artefatos: %s/plots/09_live_forecast.png", run_dir)
    logger.info("=" * 60)
    return {"run_dir": run_dir, "result": result}


def run_pipeline(config: dict, retrain: bool = False) -> dict | None:
    """Executa o pipeline completo ou a inferência com modelo existente.

    Se um modelo já estiver treinado e ``retrain`` for False, exibe um menu
    interativo [1/2/3] para o usuário escolher a ação. Com ``retrain=True``
    sobrescreve artefatos sem perguntar.

    Args:
        config: dicionário de configuração.
        retrain: se True, força novo treino mesmo que modelo exista.

    Returns:
        Dicionário com ``run_dir``, ``metrics`` e ``diagnostico``,
        ou None se o usuário cancelar (opção 3).

    Raises:
        Exception: propaga erros de qualquer etapa com contexto no log.
    """
    t_start = time.time()
    run_dir = build_run_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    _banner(config, run_dir)

    # --- Detecção de modelo existente ---
    if _model_exists(run_dir) and not retrain:
        choice = _prompt_existing_model(run_dir, config)
        if choice == 3:
            logger.info("Operação cancelada pelo usuário.")
            return None
        if choice == 1:
            return _run_inference_only(config, run_dir, t_start)
        # choice == 2: treinar novamente → prossegue para o pipeline completo

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
