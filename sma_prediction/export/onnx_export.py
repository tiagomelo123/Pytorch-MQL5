"""Exportação do modelo LSTM treinado para o formato ONNX (uso no MQL5)."""

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from data.features import get_feature_columns

logger = logging.getLogger(__name__)

_OPSET = 12  # compatível com MT5 build 3683+


def export_to_onnx(
    model: nn.Module,
    config: dict,
    run_dir: str,
    scaler,
    num_features: int,
) -> str:
    """Exporta o modelo Seq2Seq treinado para ONNX e valida contra o PyTorch.

    Gera ``run_dir/model.onnx`` e ``run_dir/onnx_metadata.json``. A validação
    compara a saída do ONNX Runtime com a do PyTorch para o mesmo input dummy.

    Args:
        model: modelo treinado.
        config: dicionário de configuração.
        run_dir: diretório de artefatos.
        scaler: ``MinMaxScaler`` fitado (para exportar min/max).
        num_features: número de features de entrada.

    Returns:
        Caminho do arquivo ``.onnx`` gerado.

    Raises:
        AssertionError: se a divergência ONNX↔PyTorch exceder 1e-5.
    """
    model = model.to("cpu")
    model.eval()

    lookback = config["lookback_window"]
    dummy_input = torch.zeros(1, lookback, num_features, dtype=torch.float32)

    os.makedirs(run_dir, exist_ok=True)
    onnx_path = os.path.join(run_dir, "model.onnx")
    tmp_path = onnx_path + ".tmp"

    torch.onnx.export(
        model,
        dummy_input,
        tmp_path,
        export_params=True,
        opset_version=_OPSET,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["ma_delta"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "ma_delta": {0: "batch_size"},
        },
    )

    logger.info("  📐 Input shape : (1, %d, %d)", lookback, num_features)
    logger.info("  📐 Output shape: (1, %d)", config["forecast_steps"])

    # --- Validação ONNX vs PyTorch ---
    import onnxruntime as ort

    session = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(["ma_delta"], {"features": dummy_input.numpy()})[0]
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    max_diff = float(np.abs(ort_out - pt_out).max())
    if max_diff >= 1e-5:
        os.remove(tmp_path)
        raise AssertionError(
            f"Divergência ONNX vs PyTorch alta: {max_diff:.2e} (limite 1e-5). "
            "Arquivo anterior preservado."
        )

    os.replace(tmp_path, onnx_path)
    logger.info("  ✅ Exportado  : %s", onnx_path)
    logger.info("  ✅ Validado   : divergência máxima PyTorch↔ONNX = %.2e", max_diff)

    # --- Metadados para replicar pré-processamento no MQL5 ---
    feature_cols = get_feature_columns(config["timeframe"])
    metadata = {
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "ma_period": config["ma_period"],
        "forecast_steps": config["forecast_steps"],
        "lookback_window": lookback,
        "num_features": num_features,
        "feature_order": feature_cols,
        "target": "delta_atr",
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
        "opset_version": _OPSET,
        "torch_version": torch.__version__,
        "export_date": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(run_dir, "onnx_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  💾 Metadados  : %s", meta_path)

    return onnx_path
