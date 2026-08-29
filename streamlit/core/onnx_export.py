"""Exportação de modelos treinados para ONNX, prontos para uso em MQL5.

Segue a mesma convenção usada nos demais projetos do repositório
(``sma_prediction/export/onnx_export.py``): opset 12 (compatível com
terminais MT5 recentes), validação numérica ONNX↔PyTorch e um arquivo de
metadados JSON com tudo que o Expert Advisor precisa para replicar o
pré-processamento (ordem das features, fórmulas, parâmetros do scaler).

Suporta os três modos de saída do painel (regressão, classificação binária
e classificação multi-classe), embutindo a ativação final (sigmoid/softmax)
no próprio grafo ONNX para que o EA em MQL5 não precise reimplementá-la.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from core.config import REGIME_CLASSES
from core.features import FEATURE_FORMULAS

OPSET_VERSION = 12  # compatível com MT5 build 3683+


def _infer_output_mode(run_config: dict) -> str:
    """Deduz o ``output_mode`` para runs antigos que não salvaram o campo."""
    if "output_mode" in run_config:
        return run_config["output_mode"]
    tarefa = run_config["tarefa"]
    if tarefa.startswith("Classificação (regime"):
        return "classificacao_multiclasse"
    if tarefa.startswith("Classificação"):
        return "classificacao_binaria"
    return "regressao"


class _ONNXWrapper(nn.Module):
    """Envolve o modelo para exportação: embute a ativação final no grafo
    ONNX (sigmoid para classificação binária, softmax para multi-classe) e
    garante saída sempre 2D ``(batch, output_size)``."""

    def __init__(self, base_model: nn.Module, activation: str) -> None:
        super().__init__()
        self.base_model = base_model
        self.activation = activation  # "sigmoid" | "softmax" | "linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        if self.activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation == "softmax":
            out = torch.softmax(out, dim=-1)
        return out


def export_to_onnx(
    model: nn.Module,
    run_config: dict,
    run_dir: str,
) -> tuple[str, dict]:
    """Exporta um modelo treinado (já com pesos carregados) para ONNX.

    Args:
        model: instância do modelo com os pesos do melhor checkpoint já
            carregados (``load_state_dict``).
        run_config: dicionário de configuração do run (mesmo formato salvo
            em ``runs/<run_id>/config.json`` pelo registry).
        run_dir: diretório onde salvar ``model.onnx`` e
            ``onnx_metadata.json``.

    Returns:
        Tupla ``(onnx_path, metadata)``.

    Raises:
        AssertionError: se a divergência numérica ONNX↔PyTorch for alta.
    """
    output_mode = _infer_output_mode(run_config)
    lookback = int(run_config["lookback"])
    feature_cols = run_config["feature_cols"]
    num_features = len(feature_cols)
    output_size = int(run_config.get("output_size") or (len(REGIME_CLASSES) if output_mode == "classificacao_multiclasse" else 1))

    activation = {
        "regressao": "linear",
        "classificacao_binaria": "sigmoid",
        "classificacao_multiclasse": "softmax",
    }[output_mode]

    model = model.to("cpu")
    model.eval()
    wrapper = _ONNXWrapper(model, activation=activation)
    wrapper.eval()

    # Input aleatório (não todo-zero) para validar melhor a equivalência numérica.
    torch.manual_seed(0)
    dummy_input = torch.randn(1, lookback, num_features, dtype=torch.float32)

    os.makedirs(run_dir, exist_ok=True)
    onnx_path = os.path.join(run_dir, "model.onnx")
    tmp_path = onnx_path + ".tmp"

    export_kwargs = dict(
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["output"],
        dynamic_axes={"features": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    try:
        # Força o exportador legado (TorchScript-based): respeita o opset
        # pedido exatamente, ao contrário do exportador "dynamo" (padrão em
        # versões recentes do PyTorch), que pode manter um opset mais novo
        # se a conversão de versão falhar — quebrando a compatibilidade com
        # terminais MT5 mais antigos. `dynamo` só existe em PyTorch mais
        # recentes; versões antigas já usam o exportador legado por padrão.
        torch.onnx.export(wrapper, dummy_input, tmp_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy_input, tmp_path, **export_kwargs)

    import onnxruntime as ort

    session = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(["output"], {"features": dummy_input.numpy()})[0]
    with torch.no_grad():
        pt_out = wrapper(dummy_input).numpy()

    max_diff = float(np.abs(ort_out - pt_out).max())
    if max_diff >= 1e-4:
        os.remove(tmp_path)
        raise AssertionError(
            f"Divergência ONNX vs PyTorch alta: {max_diff:.2e} (limite 1e-4). "
            "Arquivo anterior preservado."
        )

    os.replace(tmp_path, onnx_path)

    scaler_path = os.path.join(run_dir, "scaler.pkl")
    import joblib

    scaler = joblib.load(scaler_path)

    if output_mode == "classificacao_multiclasse":
        output_meaning = (
            f"vetor de {output_size} probabilidades (softmax) — índice "
            + ", ".join(f"{i}={nome}" for i, nome in enumerate(REGIME_CLASSES))
            + " · classe prevista = argmax do vetor"
        )
    elif output_mode == "classificacao_binaria":
        output_meaning = (
            "probabilidade (0 a 1)"
            + (
                " de o pullback continuar a tendência (>0.5 = continuação)"
                if "pullback" in run_config["tarefa"].lower()
                else " de alta no horizonte definido (>0.5 = alta)"
            )
        )
    else:
        output_meaning = f"retorno percentual previsto {run_config.get('horizon')} barras à frente"

    metadata = {
        "run_id": run_config["run_id"],
        "symbol": run_config.get("symbol"),
        "timeframe": run_config.get("timeframe"),
        "tarefa": run_config["tarefa"],
        "output_mode": output_mode,
        "arquitetura": run_config["arquitetura"],
        "lookback_window": lookback,
        "horizon": run_config.get("horizon"),
        "num_features": num_features,
        "feature_order": feature_cols,
        "feature_formulas": {c: FEATURE_FORMULAS.get(c, "ver core/features.py") for c in feature_cols},
        "scaler_type": "StandardScaler (z-score)",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "input_shape": [1, lookback, num_features],
        "output_shape": [1, output_size],
        "output_activation": activation,
        "output_meaning": output_meaning,
        "classes": REGIME_CLASSES if output_mode == "classificacao_multiclasse" else None,
        "parametros_pullback": run_config.get("parametros_pullback"),
        "parametros_regime": run_config.get("parametros_regime"),
        "opset_version": OPSET_VERSION,
        "torch_version": torch.__version__,
        "export_date": datetime.now(timezone.utc).isoformat(),
        "validacao_max_diff_onnx_pytorch": max_diff,
        "instrucoes_mql5": (
            "1) Para cada nova barra fechada, calcule as `num_features` colunas de "
            "'feature_order' na mesma ordem, usando as fórmulas de 'feature_formulas'. "
            "2) Monte as últimas `lookback_window` barras dessas features em um buffer "
            "float[lookback_window*num_features], normalizando cada valor com "
            "(valor - scaler_mean[i]) / scaler_scale[i] (i = índice da feature dentro da barra). "
            f"3) Rode o modelo ONNX com input shape [1, lookback_window, num_features] e leia "
            f"'output' (shape [1, {output_size}]). 4) Aplique 'output_meaning' para interpretar o "
            "resultado (para multi-classe, pegue o índice de maior valor)."
        ),
    }

    meta_path = os.path.join(run_dir, "onnx_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return onnx_path, metadata
