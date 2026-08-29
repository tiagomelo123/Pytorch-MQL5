"""Página: compara métricas e curvas de perda entre os modelos já treinados."""

import json
import os

import pandas as pd
import streamlit as st

from core import registry
from core.config import RUNS_DIR

st.set_page_config(page_title="Comparar Modelos", page_icon="📊", layout="wide")
st.title("📊 Comparar Modelos Treinados")

df_runs = registry.list_runs()

if df_runs.empty:
    st.info("Nenhum modelo treinado ainda. Vá em **Treinar Modelo**.")
    st.stop()

st.subheader("Todos os runs")
colunas_exibir = [
    c for c in [
        "run_id", "criado_em", "symbol", "timeframe", "tarefa", "arquitetura",
        "lookback", "horizon", "melhor_epoca", "metrica_principal", "valor_metrica",
        "enviado_firebase",
    ] if c in df_runs.columns
]
st.dataframe(df_runs[colunas_exibir], use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    filtro_symbol = st.multiselect("Filtrar por símbolo", sorted(df_runs["symbol"].dropna().unique()))
with c2:
    filtro_arq = st.multiselect("Filtrar por arquitetura", sorted(df_runs["arquitetura"].dropna().unique()))

df_filtrado = df_runs.copy()
if filtro_symbol:
    df_filtrado = df_filtrado[df_filtrado["symbol"].isin(filtro_symbol)]
if filtro_arq:
    df_filtrado = df_filtrado[df_filtrado["arquitetura"].isin(filtro_arq)]

st.subheader("Comparar métrica principal")
if not df_filtrado.empty:
    st.bar_chart(df_filtrado.set_index("run_id")["valor_metrica"], height=300)

st.subheader("Comparar curvas de perda (validação)")
opcoes_runs = df_filtrado["run_id"].tolist()
selecionados = st.multiselect("Escolha até 5 runs para sobrepor", opcoes_runs, max_selections=5)

if selecionados:
    curvas = {}
    for run_id in selecionados:
        caminho = os.path.join(RUNS_DIR, run_id, "loss_history.json")
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                hist = json.load(f)
            serie = pd.DataFrame(hist).set_index("epoch")["val_loss"]
            curvas[run_id] = serie
    if curvas:
        df_curvas = pd.DataFrame(curvas)
        st.line_chart(df_curvas, height=350)
    else:
        st.warning("Histórico de perda não encontrado localmente para os runs selecionados.")

st.divider()
st.subheader("Detalhes e ações por run")
run_escolhido = st.selectbox("Selecione um run", opcoes_runs if opcoes_runs else df_runs["run_id"].tolist())
if run_escolhido:
    run = registry.get_run(run_escolhido)
    if run:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.json({k: v for k, v in run.items() if k != "metricas_teste"}, expanded=False)
            if "metricas_teste" in run:
                st.write("**Métricas de teste:**", run["metricas_teste"])
        with c2:
            if st.button("🗑️ Remover do registro", key=f"del_run_{run_escolhido}"):
                registry.delete_run(run_escolhido)
                st.rerun()
            st.caption("Isso remove apenas a entrada do registro, não os arquivos locais em `runs/`.")
