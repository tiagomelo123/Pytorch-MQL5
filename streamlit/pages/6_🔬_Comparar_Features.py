"""Página: treina várias combinações de features (mesma tarefa/hiperparâmetros)
e compara os resultados, para ajudar a decidir quais features valem a pena."""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from core import dataset as ds
from core import features as feat
from core import labeling, models, registry, train
from core.config import (
    ARQUITETURAS,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EMA_FAST,
    DEFAULT_EMA_SLOW,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_HORIZON,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOOKBACK_WINDOW,
    DEFAULT_MIN_RETRACEMENT,
    DEFAULT_MR_ADX_MAX,
    DEFAULT_MR_ATR_PERIOD,
    DEFAULT_MR_HORIZON,
    DEFAULT_MR_SL_ATR_MULT,
    DEFAULT_MR_TP_ATR_MULT,
    DEFAULT_MR_USE_ADX_FILTER,
    DEFAULT_MR_ZSCORE_THRESHOLD,
    DEFAULT_MR_ZSCORE_WINDOW,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PATIENCE,
    DEFAULT_PULLBACK_HORIZON,
    DEFAULT_REGIME_HORIZON,
    DEFAULT_REGIME_K_LATERAL,
    DEFAULT_REGIME_VOL_WINDOW,
    DEFAULT_SEED,
    DEFAULT_SWING_ORDER,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_WEIGHT_DECAY,
    REGIME_CLASSES,
    TAREFA_DESCRICOES,
    TAREFAS,
)

st.set_page_config(page_title="Comparar Features", page_icon="🔬", layout="wide")
st.title("🔬 Comparar Combinações de Features")
st.caption(
    "Treina um modelo para cada combinação de features (mesma tarefa, arquitetura e "
    "hiperparâmetros) e compara as métricas de teste lado a lado — uma forma empírica "
    "de descobrir quantas e quais features realmente ajudam, em vez de adivinhar."
)

os.makedirs(DATA_DIR, exist_ok=True)
arquivos_locais = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
if not arquivos_locais:
    st.warning("Nenhum dataset em cache local. Vá em **Exportar Dados** ou **Datasets** primeiro.")
    st.stop()

st.subheader("1. Dataset e tarefa")
c1, c2 = st.columns(2)
with c1:
    nome_dataset = st.selectbox("Dataset (cache local)", arquivos_locais)
    df_raw = pd.read_csv(os.path.join(DATA_DIR, nome_dataset), parse_dates=["time"])
    st.caption(f"{len(df_raw)} barras · {df_raw['time'].iloc[0]} a {df_raw['time'].iloc[-1]}")
with c2:
    tarefa = st.selectbox("Tarefa", TAREFAS)
    st.caption(TAREFA_DESCRICOES.get(tarefa, ""))

is_pullback = tarefa.startswith("Classificação (pullback")
is_regime = tarefa.startswith("Classificação (regime")
is_mean_reversal = tarefa.startswith("Classificação (reversão à média")
is_classificacao = tarefa.startswith("Classificação")
output_mode = "classificacao_multiclasse" if is_regime else ("classificacao_binaria" if is_classificacao else "regressao")
maior_e_melhor = output_mode != "regressao"

c1, c2 = st.columns(2)
if is_pullback:
    with c1:
        ema_fast = st.number_input("EMA rápida", min_value=2, max_value=200, value=DEFAULT_EMA_FAST)
        ema_slow = st.number_input("EMA lenta", min_value=5, max_value=400, value=DEFAULT_EMA_SLOW)
        swing_order = st.number_input("Barras p/ confirmar swing", min_value=2, max_value=50, value=DEFAULT_SWING_ORDER)
    with c2:
        pullback_horizon = st.number_input("Barras à frente p/ checar continuação", min_value=2, max_value=300, value=DEFAULT_PULLBACK_HORIZON)
        min_retracement = st.number_input(
            "Retração mínima (%)", min_value=0.0, max_value=10.0, value=DEFAULT_MIN_RETRACEMENT * 100, step=0.05, format="%.3f"
        ) / 100
        lookback = st.number_input("Lookback", min_value=5, max_value=500, value=DEFAULT_LOOKBACK_WINDOW)
    horizon = pullback_horizon
elif is_regime:
    with c1:
        regime_horizon = st.number_input("Barras à frente p/ definir o regime", min_value=2, max_value=300, value=DEFAULT_REGIME_HORIZON)
        vol_window = st.number_input("Janela de volatilidade", min_value=5, max_value=200, value=DEFAULT_REGIME_VOL_WINDOW)
    with c2:
        k_lateral = st.number_input("Multiplicador do limiar lateral (k)", min_value=0.1, max_value=5.0, value=DEFAULT_REGIME_K_LATERAL, step=0.1)
        lookback = st.number_input("Lookback", min_value=5, max_value=500, value=DEFAULT_LOOKBACK_WINDOW)
    horizon = regime_horizon
elif is_mean_reversal:
    with c1:
        zscore_window = st.number_input("Janela do z-score", min_value=5, max_value=200, value=DEFAULT_MR_ZSCORE_WINDOW)
        zscore_threshold = st.number_input(
            "Limiar de z-score", min_value=0.5, max_value=5.0, value=DEFAULT_MR_ZSCORE_THRESHOLD, step=0.1
        )
        use_adx_filter = st.checkbox("Filtrar por ADX", value=DEFAULT_MR_USE_ADX_FILTER)
        adx_max = st.number_input(
            "ADX máximo", min_value=5.0, max_value=60.0, value=DEFAULT_MR_ADX_MAX, step=1.0, disabled=not use_adx_filter
        )
    with c2:
        tp_atr_mult = st.number_input("TP (× ATR)", min_value=0.1, max_value=10.0, value=DEFAULT_MR_TP_ATR_MULT, step=0.1)
        sl_atr_mult = st.number_input("SL (× ATR)", min_value=0.1, max_value=10.0, value=DEFAULT_MR_SL_ATR_MULT, step=0.1)
        mr_horizon = st.number_input("Barras à frente p/ checar TP/SL", min_value=2, max_value=300, value=DEFAULT_MR_HORIZON)
        lookback = st.number_input("Lookback", min_value=5, max_value=500, value=DEFAULT_LOOKBACK_WINDOW)
    atr_period = DEFAULT_MR_ATR_PERIOD
    horizon = mr_horizon
else:
    with c1:
        horizon = st.number_input("Horizonte de previsão (barras à frente)", min_value=1, max_value=200, value=DEFAULT_HORIZON)
    with c2:
        lookback = st.number_input("Lookback", min_value=5, max_value=500, value=DEFAULT_LOOKBACK_WINDOW)

st.subheader("2. Hiperparâmetros (iguais para todas as combinações)")
c1, c2, c3 = st.columns(3)
with c1:
    arquitetura = st.selectbox("Arquitetura", ARQUITETURAS)
    hidden_size = st.number_input("Camada oculta", min_value=8, max_value=1024, value=DEFAULT_HIDDEN_SIZE, step=8)
    num_layers = st.number_input("Camadas (LSTM/GRU)", min_value=1, max_value=6, value=DEFAULT_NUM_LAYERS)
    dropout = st.slider("Dropout", 0.0, 0.8, DEFAULT_DROPOUT)
with c2:
    epochs = st.number_input(
        "Épocas", min_value=1, max_value=2000, value=30,
        help="Para comparação rápida entre combinações, vale usar menos épocas do que num treino final.",
    )
    batch_size = st.number_input("Batch size", min_value=8, max_value=2048, value=DEFAULT_BATCH_SIZE, step=8)
    patience = st.number_input("Paciência (early stopping)", min_value=1, max_value=200, value=8)
with c3:
    learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=DEFAULT_LEARNING_RATE, format="%.5f")
    weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHT_DECAY, format="%.6f")
    train_ratio = st.slider("Proporção treino", 0.5, 0.9, DEFAULT_TRAIN_RATIO)
    val_ratio = st.slider("Proporção validação", 0.05, 0.4, DEFAULT_VAL_RATIO)

st.subheader("3. Pool de features candidatas")
if is_pullback:
    pool_default = feat.FEATURES_PULLBACK_SUGERIDAS
elif is_regime:
    pool_default = feat.FEATURES_REGIME_SUGERIDAS
elif is_mean_reversal:
    pool_default = feat.FEATURES_MEAN_REVERSAL_SUGERIDAS
else:
    pool_default = list(feat.FEATURES_DISPONIVEIS.keys())[:8]

nomes_pool = st.multiselect(
    "Features candidatas (universo de onde as combinações serão montadas)",
    options=list(feat.FEATURES_DISPONIVEIS.keys()),
    default=pool_default,
    key=f"pool_{tarefa}",
)
pool_keys = [feat.FEATURES_DISPONIVEIS[n] for n in nomes_pool]
if not is_pullback and "retracao_pct" in pool_keys:
    st.warning("A feature 'Retração vs. último swing (%)' só se aplica à tarefa de pullback — será ignorada.")
    pool_keys = [k for k in pool_keys if k != "retracao_pct"]
    nomes_pool = [n for n in nomes_pool if feat.FEATURES_DISPONIVEIS[n] != "retracao_pct"]

st.subheader("4. Combinações a testar")
modo = st.radio(
    "Modo de comparação",
    [
        "Combinações manuais",
        "Cada feature sozinha (ablação individual)",
        "Remover uma de cada vez (leave-one-out)",
    ],
    help=(
        "Manual: você monta cada combinação. Ablação individual: testa cada feature do pool "
        "isoladamente, para ver o poder preditivo de cada uma sozinha. Leave-one-out: testa o pool "
        "inteiro menos uma feature de cada vez, para ver o impacto de remover cada uma."
    ),
)
incluir_baseline = st.checkbox("Incluir combinação com todas as features do pool (referência)", value=True)

combinacoes: list[dict] = []

if modo == "Combinações manuais":
    if "combos_comparacao" not in st.session_state:
        st.session_state["combos_comparacao"] = []

    with st.form("form_add_combo", clear_on_submit=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            nome_combo = st.text_input("Nome da combinação", placeholder="ex.: só_momentum")
        with c2:
            features_combo = st.multiselect("Features desta combinação", options=nomes_pool)
        add = st.form_submit_button("➕ Adicionar combinação")
        if add:
            if not nome_combo.strip():
                st.warning("Dê um nome para a combinação.")
            elif not features_combo:
                st.warning("Selecione ao menos uma feature.")
            else:
                st.session_state["combos_comparacao"].append(
                    {"nome": nome_combo.strip(), "features": [feat.FEATURES_DISPONIVEIS[n] for n in features_combo]}
                )

    if st.session_state["combos_comparacao"]:
        st.write("Combinações adicionadas:")
        for i, combo in enumerate(st.session_state["combos_comparacao"]):
            c1, c2 = st.columns([5, 1])
            c1.write(f"**{combo['nome']}** — {len(combo['features'])} features: {', '.join(combo['features'])}")
            if c2.button("🗑️ Remover", key=f"rm_combo_{i}"):
                st.session_state["combos_comparacao"].pop(i)
                st.rerun()
        if st.button("🧹 Limpar todas as combinações"):
            st.session_state["combos_comparacao"] = []
            st.rerun()

    combinacoes = list(st.session_state["combos_comparacao"])

elif modo == "Cada feature sozinha (ablação individual)":
    combinacoes = [{"nome": nome, "features": [key]} for nome, key in zip(nomes_pool, pool_keys)]

else:  # leave-one-out
    combinacoes = [
        {"nome": f"sem_{nome}", "features": [k for k in pool_keys if k != key]}
        for nome, key in zip(nomes_pool, pool_keys)
        if len(pool_keys) > 1
    ]

if incluir_baseline and pool_keys:
    combinacoes = [{"nome": "Todas (baseline)", "features": list(pool_keys)}] + combinacoes

if combinacoes:
    st.caption(f"{len(combinacoes)} combinação(ões) serão treinadas nesta rodada.")

st.divider()
rodar = st.button("🚀 Rodar comparação", type="primary", disabled=(len(combinacoes) < 1))
if len(combinacoes) < 1:
    st.info("Adicione pelo menos uma combinação de features para habilitar a comparação.")

if rodar:
    # Rótulos (pullback/regime) não dependem do conjunto de features — calculados uma única vez.
    labels_df = None
    with st.spinner("Preparando rótulos..."):
        if is_pullback:
            labels_df = labeling.build_pullback_dataset(
                df_raw, ema_fast=int(ema_fast), ema_slow=int(ema_slow), swing_order=int(swing_order),
                horizon=int(pullback_horizon), min_retracement=float(min_retracement),
            )
            if labels_df["label"].notna().sum() == 0:
                st.error("Nenhum candidato a pullback encontrado com esses parâmetros.")
                st.stop()
        elif is_regime:
            labels_df = labeling.build_market_regime_labels(
                df_raw, horizon=int(regime_horizon), vol_window=int(vol_window), k_lateral=float(k_lateral)
            )
            if labels_df["label"].notna().sum() < 200:
                st.error("Poucas barras rotuladas com esses parâmetros.")
                st.stop()
        elif is_mean_reversal:
            labels_df = labeling.build_mean_reversal_dataset(
                df_raw, zscore_window=int(zscore_window), zscore_threshold=float(zscore_threshold),
                use_adx_filter=bool(use_adx_filter), adx_max=float(adx_max), atr_period=int(atr_period),
                tp_atr_mult=float(tp_atr_mult), sl_atr_mult=float(sl_atr_mult), horizon=int(mr_horizon),
            )
            if labels_df["label"].notna().sum() == 0:
                st.error("Nenhum candidato a reversão à média encontrado com esses parâmetros.")
                st.stop()

    resultados = []
    modelos_treinados = {}
    barra_geral = st.progress(0.0, text="Iniciando comparação...")

    for i, combo in enumerate(combinacoes):
        nome_combo = combo["nome"]
        feature_keys = combo["features"]
        barra_geral.progress(i / len(combinacoes), text=f"Treinando combinação {i + 1}/{len(combinacoes)}: {nome_combo}")

        try:
            if is_pullback:
                features_df = feat.build_features(df_raw, feature_keys, context=labels_df)
            else:
                features_df = feat.build_features(df_raw, feature_keys)
            feature_cols = feat.feature_columns(features_df)
            if not feature_cols:
                resultados.append({"combinacao": nome_combo, "n_features": 0, "erro": "nenhuma feature válida"})
                continue

            if is_pullback or is_regime or is_mean_reversal:
                X, y, _ = ds.build_windows_labeled(features_df, feature_cols, labels_df, int(lookback))
            else:
                X, y, _ = ds.build_windows(features_df, feature_cols, int(lookback), int(horizon), tarefa)

            if len(X) < 200:
                resultados.append({"combinacao": nome_combo, "n_features": len(feature_cols), "erro": f"poucas amostras ({len(X)})"})
                continue

            splits = ds.split_chronological(X, y, train_ratio, val_ratio)
            splits_scaled, scaler = ds.scale_splits(splits)
            loaders = ds.make_loaders(splits_scaled, int(batch_size), y_long=is_regime)

            pos_weight, class_weights, output_size = None, None, 1
            if output_mode == "classificacao_binaria":
                y_train = splits["train"][1]
                n_pos, n_neg = float((y_train == 1).sum()), float((y_train == 0).sum())
                if n_pos > 0 and n_neg > 0:
                    pos_weight = n_neg / n_pos
            elif output_mode == "classificacao_multiclasse":
                output_size = len(REGIME_CLASSES)
                y_train = splits["train"][1]
                contagens = np.array([max(1, (y_train == c).sum()) for c in range(output_size)], dtype=float)
                class_weights = (contagens.sum() / (output_size * contagens)).tolist()

            model = models.build_model(
                arquitetura, len(feature_cols), int(lookback), int(hidden_size), int(num_layers), dropout, output_size
            )
            config = {
                "epochs": int(epochs), "learning_rate": float(learning_rate), "weight_decay": float(weight_decay),
                "patience": int(patience), "seed": DEFAULT_SEED,
            }
            resultado = train.train_model(
                model, loaders, config, output_mode, pos_weight=pos_weight, class_weights=class_weights
            )
            metricas = train.evaluate(model, loaders["test"], output_mode)

            if output_mode == "classificacao_multiclasse":
                metrica_principal, valor_metrica = "acuracia", metricas["acuracia"]
                extra = f"F1 macro={metricas['f1_macro']:.3f}"
            elif output_mode == "classificacao_binaria":
                metrica_principal, valor_metrica = "acuracia", metricas["acuracia"]
                extra = f"F1={metricas['f1']:.3f}"
            else:
                metrica_principal, valor_metrica = "mae", metricas["mae"]
                extra = f"R²={metricas['r2']:.3f}"

            resultados.append(
                {
                    "combinacao": nome_combo,
                    "n_features": len(feature_cols),
                    "features": ", ".join(feature_cols),
                    metrica_principal: valor_metrica,
                    "detalhe": extra,
                    "melhor_epoca": resultado["best_epoch"],
                    "erro": None,
                }
            )
            modelos_treinados[nome_combo] = {
                "best_state": resultado["best_state"],
                "best_epoch": resultado["best_epoch"],
                "scaler": scaler,
                "feature_cols": feature_cols,
                "output_size": output_size,
                "loss_history": resultado["loss_history"],
                "metricas_teste": metricas,
                "metrica_principal": metrica_principal,
                "valor_metrica": valor_metrica,
            }
        except Exception as e:  # noqa: BLE001
            resultados.append({"combinacao": nome_combo, "n_features": len(feature_keys), "erro": str(e)})

    barra_geral.progress(1.0, text="Comparação concluída.")
    st.session_state["resultados_comparacao"] = resultados
    st.session_state["modelos_comparacao"] = modelos_treinados
    st.session_state["comparacao_meta"] = {
        "tarefa": tarefa, "output_mode": output_mode, "arquitetura": arquitetura,
        "hidden_size": int(hidden_size), "num_layers": int(num_layers), "dropout": dropout,
        "lookback": int(lookback), "horizon": int(horizon), "dataset": nome_dataset,
        "metrica_principal": "mae" if output_mode == "regressao" else "acuracia",
        "maior_e_melhor": maior_e_melhor,
    }
    if is_pullback:
        st.session_state["comparacao_meta"]["parametros_pullback"] = {
            "ema_fast": int(ema_fast), "ema_slow": int(ema_slow), "swing_order": int(swing_order),
            "pullback_horizon": int(pullback_horizon), "min_retracement": float(min_retracement),
        }
    if is_regime:
        st.session_state["comparacao_meta"]["parametros_regime"] = {
            "regime_horizon": int(regime_horizon), "vol_window": int(vol_window), "k_lateral": float(k_lateral),
            "classes": REGIME_CLASSES,
        }
    if is_mean_reversal:
        st.session_state["comparacao_meta"]["parametros_reversao_media"] = {
            "zscore_window": int(zscore_window), "zscore_threshold": float(zscore_threshold),
            "use_adx_filter": bool(use_adx_filter), "adx_max": float(adx_max), "atr_period": int(atr_period),
            "tp_atr_mult": float(tp_atr_mult), "sl_atr_mult": float(sl_atr_mult), "mr_horizon": int(mr_horizon),
        }

if "resultados_comparacao" in st.session_state:
    st.subheader("Resultados")
    df_res = pd.DataFrame(st.session_state["resultados_comparacao"])
    meta = st.session_state["comparacao_meta"]
    metrica_col = meta["metrica_principal"]

    df_ok = df_res[df_res["erro"].isna()].copy() if "erro" in df_res.columns else df_res.copy()
    df_erro = df_res[df_res["erro"].notna()].copy() if "erro" in df_res.columns else pd.DataFrame()

    if not df_ok.empty:
        df_ok = df_ok.sort_values(metrica_col, ascending=not meta["maior_e_melhor"]).reset_index(drop=True)
        colunas = [c for c in ["combinacao", "n_features", metrica_col, "detalhe", "melhor_epoca", "features"] if c in df_ok.columns]
        st.dataframe(df_ok[colunas], use_container_width=True, hide_index=True)
        st.bar_chart(df_ok.set_index("combinacao")[metrica_col], height=300)

        melhor = df_ok.iloc[0]
        st.success(
            f"🏆 Melhor combinação: **{melhor['combinacao']}** ({melhor['n_features']} features) — "
            f"{metrica_col} = {melhor[metrica_col]:.4f}"
        )
        st.caption(
            "Dica: esta comparação usa menos épocas para ser rápida. Depois de escolher a melhor "
            "combinação, treine-a de novo em **Treinar Modelo** com mais épocas para o modelo final."
        )

    if not df_erro.empty:
        st.warning("Combinações com erro (não entraram na comparação):")
        st.dataframe(df_erro[["combinacao", "erro"]], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Salvar uma combinação como modelo treinado")
    nomes_disponiveis = list(st.session_state.get("modelos_comparacao", {}).keys())
    if nomes_disponiveis:
        escolha = st.selectbox("Escolha a combinação para salvar", nomes_disponiveis)
        if st.button("💾 Salvar modelo desta combinação"):
            m = st.session_state["modelos_comparacao"][escolha]
            symbol_nome = meta["dataset"].split("_")[0]
            run_id = registry.new_run_id(symbol_nome, meta["arquitetura"])
            pasta = registry.run_dir(run_id)

            torch.save(m["best_state"], os.path.join(pasta, "model.pt"))
            joblib.dump(m["scaler"], os.path.join(pasta, "scaler.pkl"))
            pd.DataFrame(m["loss_history"]).to_json(os.path.join(pasta, "loss_history.json"), orient="records", indent=2)

            metricas_serializaveis = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in m["metricas_teste"].items()
                if k not in ("predicoes", "reais", "probabilidades")
            }
            run_config = {
                "run_id": run_id,
                "criado_em": datetime.now().isoformat(timespec="seconds"),
                "dataset": meta["dataset"],
                "symbol": symbol_nome,
                "timeframe": meta["dataset"].split("_")[1] if "_" in meta["dataset"] else "",
                "tarefa": meta["tarefa"],
                "output_mode": meta["output_mode"],
                "output_size": m["output_size"],
                "arquitetura": meta["arquitetura"],
                "feature_cols": m["feature_cols"],
                "lookback": meta["lookback"],
                "horizon": meta["horizon"],
                "hidden_size": meta["hidden_size"],
                "num_layers": meta["num_layers"],
                "dropout": meta["dropout"],
                "epochs_treinadas": len(m["loss_history"]),
                "melhor_epoca": m["best_epoch"],
                "metrica_principal": m["metrica_principal"],
                "valor_metrica": m["valor_metrica"],
                "metricas_teste": metricas_serializaveis,
                "enviado_firebase": False,
                "origem": f"Comparar Features — combinação '{escolha}'",
            }
            if "parametros_pullback" in meta:
                run_config["parametros_pullback"] = meta["parametros_pullback"]
            if "parametros_regime" in meta:
                run_config["parametros_regime"] = meta["parametros_regime"]
            if "parametros_reversao_media" in meta:
                run_config["parametros_reversao_media"] = meta["parametros_reversao_media"]

            with open(os.path.join(pasta, "config.json"), "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=2, ensure_ascii=False)
            registry.add_run(run_config)
            st.success(f"Modelo salvo em `runs/{run_id}/` — já aparece em Comparar Modelos e Exportar ONNX.")
    else:
        st.caption("Nenhuma combinação treinada com sucesso para salvar.")
