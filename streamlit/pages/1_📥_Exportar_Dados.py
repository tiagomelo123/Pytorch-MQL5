"""Página: conecta ao MT5 e exporta dataseries OHLCV, com envio ao Firebase."""

import datetime as dt
import os

import streamlit as st

from core import firebase_client, mt5_data
from core.config import (
    DATA_DIR,
    DEFAULT_BARS_HISTORY,
    FIREBASE_DATASETS_PREFIX,
    SYMBOLS_SUGERIDOS,
    TIMEFRAMES,
)

st.set_page_config(page_title="Exportar Dados MT5", page_icon="📥", layout="wide")
st.title("📥 Exportar Dataseries do MetaTrader 5")

if not mt5_data.is_available():
    st.error(
        "O pacote `MetaTrader5` não está instalado nesta máquina, ou o painel "
        "não está rodando em um Windows com o terminal MT5 instalado. "
        "Rode `pip install MetaTrader5` e abra o terminal MT5 logado em uma conta."
    )

with st.form("form_exportacao"):
    c1, c2 = st.columns(2)
    with c1:
        symbol = st.selectbox(
            "Símbolo", options=SYMBOLS_SUGERIDOS, index=0,
            help="Selecione um símbolo sugerido ou digite outro no campo abaixo.",
        )
        symbol_manual = st.text_input("Ou digite outro símbolo", value="")
        symbol_final = symbol_manual.strip().upper() or symbol
    with c2:
        timeframe = st.selectbox("Timeframe", options=TIMEFRAMES, index=TIMEFRAMES.index("H1"))

    modo = st.radio("Modo de coleta", ["Quantidade de barras", "Intervalo de datas"], horizontal=True)

    if modo == "Quantidade de barras":
        bars = st.number_input("Quantidade de barras", min_value=100, max_value=200000, value=DEFAULT_BARS_HISTORY, step=100)
        data_inicio = data_fim = None
    else:
        bars = DEFAULT_BARS_HISTORY
        colA, colB = st.columns(2)
        with colA:
            data_inicio = st.date_input("Data inicial", value=dt.date.today() - dt.timedelta(days=180))
        with colB:
            data_fim = st.date_input("Data final", value=dt.date.today())

    enviado = st.form_submit_button("🔗 Conectar ao MT5 e exportar", type="primary")

if enviado:
    modo_interno = "barras" if modo == "Quantidade de barras" else "periodo"
    with st.spinner(f"Conectando ao MT5 e baixando {symbol_final} {timeframe}..."):
        try:
            df = mt5_data.export_ohlcv(
                symbol=symbol_final,
                timeframe=timeframe,
                modo=modo_interno,
                bars=int(bars),
                data_inicio=data_inicio,
                data_fim=data_fim,
            )
            st.session_state["ultimo_dataset"] = df
            st.session_state["ultimo_dataset_nome"] = f"{symbol_final}_{timeframe}"
            st.success(f"{len(df)} barras exportadas: {df['time'].iloc[0]} a {df['time'].iloc[-1]}")
        except mt5_data.MT5Error as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro inesperado: {e}")

if "ultimo_dataset" in st.session_state:
    df = st.session_state["ultimo_dataset"]
    nome_base = st.session_state["ultimo_dataset_nome"]

    st.subheader("Prévia")
    st.line_chart(df.set_index("time")["close"], height=250)
    st.dataframe(df.tail(20), use_container_width=True, hide_index=True)

    st.subheader("Salvar dataset")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("💾 Salvar em cache local"):
            os.makedirs(DATA_DIR, exist_ok=True)
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"{nome_base}_{ts}.csv"
            caminho = os.path.join(DATA_DIR, nome_arquivo)
            df.to_csv(caminho, index=False)
            st.success(f"Salvo em cache local: {nome_arquivo}")

    with c2:
        if st.button("☁️ Enviar para Firebase Storage"):
            if not firebase_client.is_configured():
                st.error("Firebase não configurado. Veja `.streamlit/secrets.toml.example`.")
            else:
                with st.spinner("Enviando para o Firebase Storage..."):
                    try:
                        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        remote_path = f"{FIREBASE_DATASETS_PREFIX}{nome_base}_{ts}.csv"
                        firebase_client.upload_dataframe(df, remote_path)
                        st.success(f"Enviado para o Firebase: {remote_path}")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Falha ao enviar para o Firebase: {e}")
