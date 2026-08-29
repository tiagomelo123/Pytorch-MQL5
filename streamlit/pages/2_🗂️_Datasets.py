"""Página: gerencia os datasets salvos localmente e no Firebase Storage."""

import os

import pandas as pd
import streamlit as st

from core import firebase_client
from core.config import DATA_DIR, FIREBASE_DATASETS_PREFIX

st.set_page_config(page_title="Datasets", page_icon="🗂️", layout="wide")
st.title("🗂️ Datasets")

aba_local, aba_firebase = st.tabs(["💻 Cache local", "☁️ Firebase Storage"])

with aba_local:
    os.makedirs(DATA_DIR, exist_ok=True)
    arquivos = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    if not arquivos:
        st.info("Nenhum dataset em cache local ainda. Use a página **Exportar Dados**.")
    else:
        for nome in arquivos:
            caminho = os.path.join(DATA_DIR, nome)
            tamanho_kb = round(os.path.getsize(caminho) / 1024, 1)
            with st.expander(f"📄 {nome}  ·  {tamanho_kb} KB"):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    if st.button("👁️ Pré-visualizar", key=f"prev_{nome}"):
                        df_prev = pd.read_csv(caminho, parse_dates=["time"])
                        st.line_chart(df_prev.set_index("time")["close"], height=200)
                        st.dataframe(df_prev.describe(), use_container_width=True)
                with c2:
                    if firebase_client.is_configured() and st.button("☁️ Enviar ao Firebase", key=f"up_{nome}"):
                        with st.spinner("Enviando..."):
                            try:
                                df_up = pd.read_csv(caminho)
                                remote_path = f"{FIREBASE_DATASETS_PREFIX}{nome}"
                                firebase_client.upload_file(caminho, remote_path)
                                st.success(f"Enviado: {remote_path}")
                            except Exception as e:  # noqa: BLE001
                                st.error(f"Erro: {e}")
                with c3:
                    if st.button("🗑️ Excluir", key=f"del_{nome}"):
                        os.remove(caminho)
                        st.rerun()

with aba_firebase:
    if not firebase_client.is_configured():
        st.warning("Firebase não configurado. Veja `.streamlit/secrets.toml.example`.")
    else:
        if st.button("🔄 Atualizar lista"):
            st.cache_data.clear()

        @st.cache_data(ttl=30, show_spinner="Listando datasets no Firebase...")
        def _listar():
            return firebase_client.list_blobs(FIREBASE_DATASETS_PREFIX)

        try:
            blobs = _listar()
        except Exception as e:  # noqa: BLE001
            blobs = []
            st.error(f"Erro ao listar Firebase: {e}")

        if not blobs:
            st.info("Nenhum dataset encontrado no Firebase Storage.")
        else:
            for b in blobs:
                with st.expander(f"☁️ {b['nome']}  ·  {b['tamanho_kb']} KB  ·  {b['atualizado']}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("⬇️ Baixar para cache local", key=f"dl_{b['nome']}"):
                            destino = os.path.join(DATA_DIR, os.path.basename(b["nome"]))
                            with st.spinner("Baixando..."):
                                firebase_client.download_to_file(b["nome"], destino)
                            st.success(f"Baixado para cache local: {os.path.basename(b['nome'])}")
                    with c2:
                        if st.button("🗑️ Excluir do Firebase", key=f"rm_{b['nome']}"):
                            firebase_client.delete_blob(b["nome"])
                            st.cache_data.clear()
                            st.rerun()
