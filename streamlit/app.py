"""Painel de Gerenciamento de Redes Neurais e Dataseries do MetaTrader 5.

Página inicial: status das conexões (MT5 e Firebase) e navegação para as
demais páginas do painel (exportação de dados, datasets, treino e
comparação de modelos).
"""

import pandas as pd
import streamlit as st

from core import firebase_client, mt5_data, registry
from core.config import DATA_DIR

st.set_page_config(
    page_title="Painel Redes Neurais MT5",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Painel de Redes Neurais para Trading (MetaTrader 5)")
st.caption(
    "Exporte dataseries do MT5, treine e compare redes neurais em PyTorch, "
    "tudo em um só lugar."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔌 MetaTrader 5")
    if not mt5_data.is_available():
        st.warning("Pacote MetaTrader5 não instalado nesta máquina.")
    else:
        st.success("Pacote MetaTrader5 disponível.")
        st.caption("A conexão é feita sob demanda, ao exportar dados.")

with col2:
    st.subheader("☁️ Firebase Storage")
    if firebase_client.is_configured():
        st.success("Credenciais configuradas (st.secrets).")
    else:
        st.warning("Não configurado. Veja `.streamlit/secrets.toml.example`.")

with col3:
    st.subheader("📊 Resumo local")
    import os

    n_datasets = len([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]) if os.path.isdir(DATA_DIR) else 0
    df_runs = registry.list_runs()
    st.metric("Datasets em cache", n_datasets)
    st.metric("Modelos treinados", len(df_runs))

st.divider()

st.subheader("Fluxo de trabalho")
st.markdown(
    """
1. **📥 Exportar Dados** — conecta ao MT5, baixa a dataseries do ativo/timeframe escolhido
   e, se desejado, envia para o Firebase Storage.
2. **🗂️ Datasets** — visualiza e gerencia os datasets em cache local e no Firebase.
3. **🧠 Treinar Modelo** — escolhe features, arquitetura (LSTM/GRU/MLP) e hiperparâmetros,
   treina com acompanhamento de métricas em tempo real.
4. **📊 Comparar Modelos** — compara métricas e curvas de perda entre os modelos já treinados.

Use o menu à esquerda para navegar entre as páginas.
"""
)

if len(registry.list_runs()) > 0:
    st.subheader("Últimos modelos treinados")
    df = registry.list_runs().head(5)
    cols = [c for c in ["run_id", "criado_em", "symbol", "timeframe", "tarefa", "arquitetura", "metrica_principal", "valor_metrica"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
