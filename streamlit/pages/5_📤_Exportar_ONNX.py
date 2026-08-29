"""Página: exporta um modelo treinado (+ metadados de features) para ONNX,
pronto para uso em um Expert Advisor MQL5."""

import os

import streamlit as st
import torch

from core import firebase_client, models, onnx_export, registry
from core.config import FIREBASE_MODELS_PREFIX, RUNS_DIR

st.set_page_config(page_title="Exportar ONNX", page_icon="📤", layout="wide")
st.title("📤 Exportar Modelo para ONNX (uso em MQL5)")

st.caption(
    "Converte um modelo já treinado para o formato ONNX, que pode ser carregado "
    "diretamente em um Expert Advisor MQL5 com `OnnxCreate()`. Também gera um "
    "arquivo de metadados com a ordem das features, as fórmulas usadas e os "
    "parâmetros de normalização — necessários para replicar o pré-processamento "
    "dentro do MQL5."
)

df_runs = registry.list_runs()
if df_runs.empty:
    st.info("Nenhum modelo treinado ainda. Vá em **Treinar Modelo**.")
    st.stop()

opcoes = df_runs["run_id"].tolist()
run_id = st.selectbox("Escolha o modelo treinado", opcoes)
run_config = registry.get_run(run_id)

if run_config is None:
    st.error("Run não encontrado no registro.")
    st.stop()

pasta = registry.run_dir(run_id)
model_path = os.path.join(pasta, "model.pt")
onnx_path = os.path.join(pasta, "model.onnx")
meta_path = os.path.join(pasta, "onnx_metadata.json")

c1, c2 = st.columns(2)
with c1:
    st.write("**Símbolo/timeframe:**", f"{run_config.get('symbol')} / {run_config.get('timeframe')}")
    st.write("**Tarefa:**", run_config["tarefa"])
    st.write("**Arquitetura:**", run_config["arquitetura"])
with c2:
    st.write("**Lookback:**", run_config["lookback"])
    st.write("**Features:**", ", ".join(run_config["feature_cols"]))
    st.write("**Métrica principal:**", f"{run_config['metrica_principal']} = {run_config['valor_metrica']:.4f}")

if not os.path.exists(model_path):
    st.error(f"Arquivo `model.pt` não encontrado em `runs/{run_id}/`. Retreine este modelo.")
    st.stop()

st.divider()

if st.button("⚙️ Exportar para ONNX", type="primary"):
    with st.spinner("Reconstruindo o modelo e exportando para ONNX..."):
        try:
            model = models.build_model(
                run_config["arquitetura"],
                len(run_config["feature_cols"]),
                run_config["lookback"],
                run_config["hidden_size"],
                run_config["num_layers"],
                run_config["dropout"],
                run_config.get("output_size", 1),
            )
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)

            path_onnx, metadata = onnx_export.export_to_onnx(model, run_config, pasta)
            st.session_state["ultimo_onnx_run_id"] = run_id
            st.success(
                f"Exportado: `runs/{run_id}/model.onnx` · divergência ONNX↔PyTorch = "
                f"{metadata['validacao_max_diff_onnx_pytorch']:.2e}"
            )
        except AssertionError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro ao exportar: {e}")

if os.path.exists(onnx_path) and os.path.exists(meta_path):
    st.subheader("Arquivos gerados")

    import json

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    c1, c2 = st.columns(2)
    with c1:
        with open(onnx_path, "rb") as f:
            st.download_button(
                "⬇️ Baixar model.onnx", f.read(), file_name=f"{run_id}.onnx", mime="application/octet-stream"
            )
    with c2:
        with open(meta_path, "rb") as f:
            st.download_button(
                "⬇️ Baixar onnx_metadata.json", f.read(), file_name=f"{run_id}_onnx_metadata.json", mime="application/json"
            )

    with st.expander("📋 Ver metadados (ordem das features, fórmulas, normalização)"):
        st.json(metadata)

    st.caption(
        f"Os dois arquivos também estão salvos em `runs/{run_id}/` "
        "(`model.onnx` e `onnx_metadata.json`)."
    )

    if firebase_client.is_configured() and st.button("☁️ Enviar ONNX + metadados para o Firebase"):
        with st.spinner("Enviando ao Firebase..."):
            try:
                for arquivo in ["model.onnx", "onnx_metadata.json"]:
                    caminho_local = os.path.join(pasta, arquivo)
                    remoto = f"{FIREBASE_MODELS_PREFIX}{run_id}/{arquivo}"
                    firebase_client.upload_file(caminho_local, remoto)
                st.success("Enviado ao Firebase Storage.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Erro ao enviar: {e}")

    st.divider()
    st.subheader("Como usar no MQL5")
    st.markdown(
        f"""
1. Copie `{run_id}.onnx` para a pasta `MQL5/Files/` (ou `Common/Files/`) do seu terminal.
2. No EA, carregue o modelo com `OnnxCreate()` e defina o formato de entrada/saída com
   `OnnxSetInputShape` / `OnnxSetOutputShape` usando `input_shape` e `output_shape` do
   `onnx_metadata.json` (`{metadata['input_shape']}` e `{metadata['output_shape']}`).
3. A cada barra nova, calcule as features na ordem de `feature_order` (fórmulas em
   `feature_formulas`) e normalize cada uma com `(valor - scaler_mean[i]) / scaler_scale[i]`.
4. Monte o buffer de entrada com as últimas `{metadata['lookback_window']}` barras dessas
   features já normalizadas e rode `OnnxRun()`.
5. Interprete a saída conforme `output_meaning`: **{metadata['output_meaning']}**.
"""
    )
