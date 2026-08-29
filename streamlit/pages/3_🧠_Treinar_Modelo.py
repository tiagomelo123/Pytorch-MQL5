"""Página: configura features, arquitetura e hiperparâmetros e treina o modelo."""

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
from core import firebase_client, labeling, models, registry, train
from core.config import (
    ARQUITETURAS,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EMA_FAST,
    DEFAULT_EMA_SLOW,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_HORIZON,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOOKBACK_WINDOW,
    DEFAULT_MIN_RETRACEMENT,
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
    FIREBASE_MODELS_PREFIX,
    REGIME_CLASSES,
    TAREFA_DESCRICOES,
    TAREFAS,
)

st.set_page_config(page_title="Treinar Modelo", page_icon="🧠", layout="wide")
st.title("🧠 Treinar Rede Neural")

os.makedirs(DATA_DIR, exist_ok=True)
arquivos_locais = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

if not arquivos_locais:
    st.warning("Nenhum dataset em cache local. Vá em **Exportar Dados** ou **Datasets** primeiro.")
    st.stop()

st.subheader("1. Dataset")
nome_dataset = st.selectbox("Escolha o dataset (cache local)", arquivos_locais)
df_raw = pd.read_csv(os.path.join(DATA_DIR, nome_dataset), parse_dates=["time"])
st.caption(f"{len(df_raw)} barras · {df_raw['time'].iloc[0]} a {df_raw['time'].iloc[-1]}")

st.subheader("2. Tarefa e alvo")
tarefa = st.selectbox("Tarefa", TAREFAS)
st.info(TAREFA_DESCRICOES.get(tarefa, ""), icon="🎯")
is_pullback = tarefa.startswith("Classificação (pullback")
is_regime = tarefa.startswith("Classificação (regime")
is_classificacao = tarefa.startswith("Classificação")

if is_regime:
    output_mode = "classificacao_multiclasse"
elif is_classificacao:
    output_mode = "classificacao_binaria"
else:
    output_mode = "regressao"

c1, c2 = st.columns(2)
if is_pullback:
    with c1:
        ema_fast = st.number_input("EMA rápida (define tendência)", min_value=2, max_value=200, value=DEFAULT_EMA_FAST)
        ema_slow = st.number_input("EMA lenta (define tendência)", min_value=5, max_value=400, value=DEFAULT_EMA_SLOW)
        swing_order = st.number_input(
            "Barras de cada lado p/ confirmar swing", min_value=2, max_value=50, value=DEFAULT_SWING_ORDER,
            help="Um topo/fundo só é confirmado depois de N barras seguintes — evita usar informação futura.",
        )
    with c2:
        pullback_horizon = st.number_input(
            "Barras à frente p/ checar continuação", min_value=2, max_value=300, value=DEFAULT_PULLBACK_HORIZON
        )
        min_retracement = st.number_input(
            "Retração mínima (%) para considerar pullback", min_value=0.0, max_value=10.0,
            value=DEFAULT_MIN_RETRACEMENT * 100, step=0.05, format="%.3f",
        ) / 100
        lookback = st.number_input("Janela de contexto (lookback)", min_value=5, max_value=500, value=DEFAULT_LOOKBACK_WINDOW)
    horizon = pullback_horizon
    st.caption(
        "Rótulo: **1 = continuação** (o preço rompe o extremo do swing a favor da tendência antes de "
        "romper a estrutura contrária) · **0 = reversão/sem continuação**."
    )
elif is_regime:
    with c1:
        regime_horizon = st.number_input(
            "Barras à frente para definir o regime", min_value=2, max_value=300, value=DEFAULT_REGIME_HORIZON
        )
        vol_window = st.number_input(
            "Janela de volatilidade (barras)", min_value=5, max_value=200, value=DEFAULT_REGIME_VOL_WINDOW,
            help="Usada para estimar a volatilidade recente e calcular o limiar adaptativo de 'lateral'.",
        )
    with c2:
        k_lateral = st.number_input(
            "Multiplicador do limiar lateral (k)", min_value=0.1, max_value=5.0, value=DEFAULT_REGIME_K_LATERAL, step=0.1,
            help="Quanto maior, mais amplo o intervalo de retorno considerado 'lateral' (sem tendência clara).",
        )
        lookback = st.number_input("Janela de contexto (lookback)", min_value=5, max_value=500, value=DEFAULT_LOOKBACK_WINDOW)
    horizon = regime_horizon
    st.caption(
        f"Rótulo (3 classes): **{REGIME_CLASSES[0]}** (retorno futuro < -limiar) · "
        f"**{REGIME_CLASSES[1]}** (dentro do limiar) · **{REGIME_CLASSES[2]}** (retorno futuro > limiar). "
        "O limiar é adaptativo: `k × volatilidade recente × √horizonte`."
    )
else:
    with c1:
        horizon = st.number_input("Horizonte de previsão (barras à frente)", min_value=1, max_value=200, value=DEFAULT_HORIZON)
    with c2:
        lookback = st.number_input("Janela de contexto (lookback)", min_value=5, max_value=500, value=DEFAULT_LOOKBACK_WINDOW)

st.subheader("3. Features")
if is_pullback:
    default_features = feat.FEATURES_PULLBACK_SUGERIDAS
elif is_regime:
    default_features = feat.FEATURES_REGIME_SUGERIDAS
else:
    default_features = ["Retorno (close a close)", "Média móvel 20", "RSI 14", "Volatilidade (desvio padrão 20)"]

nomes_features = st.multiselect(
    "Features de entrada",
    options=list(feat.FEATURES_DISPONIVEIS.keys()),
    default=default_features,
    key=f"features_{tarefa}",
)
feature_keys = [feat.FEATURES_DISPONIVEIS[n] for n in nomes_features]
if not is_pullback and "retracao_pct" in feature_keys:
    st.warning("A feature 'Retração vs. último swing (%)' só está disponível na tarefa de pullback — será ignorada.")
    feature_keys = [k for k in feature_keys if k != "retracao_pct"]
if is_pullback and "retracao_pct" not in feature_keys:
    st.caption("💡 Dica: a feature 'Retração vs. último swing (%)' costuma ajudar bastante nesta tarefa.")

st.subheader("4. Arquitetura e hiperparâmetros")
c1, c2, c3 = st.columns(3)
with c1:
    arquitetura = st.selectbox("Arquitetura", ARQUITETURAS)
    hidden_size = st.number_input("Tamanho da camada oculta", min_value=8, max_value=1024, value=DEFAULT_HIDDEN_SIZE, step=8)
    num_layers = st.number_input("Número de camadas (LSTM/GRU)", min_value=1, max_value=6, value=DEFAULT_NUM_LAYERS)
    dropout = st.slider("Dropout", 0.0, 0.8, DEFAULT_DROPOUT)
with c2:
    epochs = st.number_input("Épocas", min_value=1, max_value=2000, value=DEFAULT_EPOCHS)
    batch_size = st.number_input("Batch size", min_value=8, max_value=2048, value=DEFAULT_BATCH_SIZE, step=8)
    patience = st.number_input("Paciência (early stopping)", min_value=1, max_value=200, value=DEFAULT_PATIENCE)
with c3:
    learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=DEFAULT_LEARNING_RATE, format="%.5f")
    weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=DEFAULT_WEIGHT_DECAY, format="%.6f")
    train_ratio = st.slider("Proporção treino", 0.5, 0.9, DEFAULT_TRAIN_RATIO)
    val_ratio = st.slider("Proporção validação", 0.05, 0.4, DEFAULT_VAL_RATIO)

st.divider()
iniciar = st.button("🚀 Iniciar treinamento", type="primary", disabled=(len(feature_keys) == 0))
if len(feature_keys) == 0:
    st.info("Selecione ao menos uma feature para habilitar o treino.")

if iniciar:
    labels_df = None
    with st.spinner("Construindo features e janelas..."):
        if is_pullback:
            labels_df = labeling.build_pullback_dataset(
                df_raw,
                ema_fast=int(ema_fast),
                ema_slow=int(ema_slow),
                swing_order=int(swing_order),
                horizon=int(pullback_horizon),
                min_retracement=float(min_retracement),
            )
            n_candidatos = int(labels_df["is_candidate"].sum())
            n_rotulados = int(labels_df["label"].notna().sum())
            if n_rotulados == 0:
                st.error(
                    "Nenhum candidato a pullback foi encontrado com esses parâmetros. "
                    "Tente reduzir a retração mínima, o período das EMAs ou exportar mais barras."
                )
                st.stop()
            taxa_continuacao = labels_df["label"].mean()
            st.caption(
                f"Candidatos a pullback encontrados: {n_candidatos} · rotulados: {n_rotulados} · "
                f"continuação: {taxa_continuacao:.1%} · reversão: {1 - taxa_continuacao:.1%}"
            )
            features_df = feat.build_features(df_raw, feature_keys, context=labels_df)
            feature_cols = feat.feature_columns(features_df)
            X, y, tempos = ds.build_windows_labeled(features_df, feature_cols, labels_df, int(lookback))

        elif is_regime:
            labels_df = labeling.build_market_regime_labels(
                df_raw, horizon=int(regime_horizon), vol_window=int(vol_window), k_lateral=float(k_lateral)
            )
            n_rotulados = int(labels_df["label"].notna().sum())
            if n_rotulados < 200:
                st.error(
                    f"Poucas barras rotuladas ({n_rotulados}). Exporte mais barras ou reduza o "
                    "horizonte/janela de volatilidade."
                )
                st.stop()
            contagem = labels_df["label"].value_counts().reindex([0.0, 1.0, 2.0], fill_value=0)
            dist_df = pd.DataFrame({"regime": REGIME_CLASSES, "quantidade": contagem.values})
            st.caption(f"Barras rotuladas: {n_rotulados}")
            st.bar_chart(dist_df.set_index("regime"), height=200)

            features_df = feat.build_features(df_raw, feature_keys)
            feature_cols = feat.feature_columns(features_df)
            X, y, tempos = ds.build_windows_labeled(features_df, feature_cols, labels_df, int(lookback))

        else:
            features_df = feat.build_features(df_raw, feature_keys)
            feature_cols = feat.feature_columns(features_df)
            X, y, tempos = ds.build_windows(features_df, feature_cols, int(lookback), int(horizon), tarefa)

    if len(X) < 200:
        st.error(
            f"Poucas amostras após construção das janelas ({len(X)}). Exporte mais barras, reduza o "
            "lookback/horizonte ou (nas tarefas de classificação) ajuste os parâmetros de rotulagem."
        )
        st.stop()

    splits = ds.split_chronological(X, y, train_ratio, val_ratio)
    splits_scaled, scaler = ds.scale_splits(splits)
    loaders = ds.make_loaders(splits_scaled, int(batch_size), y_long=is_regime)

    st.caption(
        f"Amostras — treino: {len(splits['train'][0])} · validação: {len(splits['val'][0])} · teste: {len(splits['test'][0])}"
    )

    pos_weight, class_weights, output_size = None, None, 1
    if output_mode == "classificacao_binaria":
        y_train = splits["train"][1]
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
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
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "patience": int(patience),
        "seed": DEFAULT_SEED,
    }

    progresso = st.progress(0.0, text="Treinando...")
    grafico_placeholder = st.empty()
    metricas_placeholder = st.empty()
    historico_ui: list[dict] = []

    def callback(epoch, total_epochs, tr_loss, val_loss, tr_metric, val_metric):
        progresso.progress(epoch / total_epochs, text=f"Época {epoch}/{total_epochs}")
        historico_ui.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})
        if epoch % max(1, total_epochs // 100) == 0 or epoch == total_epochs:
            hist_df = pd.DataFrame(historico_ui).set_index("epoch")
            grafico_placeholder.line_chart(hist_df[["train_loss", "val_loss"]], height=280)
            nome_metrica = "Acurácia" if is_classificacao else "MAE"
            metricas_placeholder.markdown(
                f"**Época {epoch}** · train_loss={tr_loss:.6f} · val_loss={val_loss:.6f} · "
                f"{nome_metrica} treino={tr_metric:.4f} · {nome_metrica} val={val_metric:.4f}"
            )

    resultado = train.train_model(
        model, loaders, config, output_mode, progress_callback=callback,
        pos_weight=pos_weight, class_weights=class_weights,
    )
    progresso.progress(1.0, text="Treino concluído.")

    metricas_teste = train.evaluate(model, loaders["test"], output_mode)

    st.success(f"Treino concluído — melhor época: {resultado['best_epoch']} (device: {resultado['device']})")

    st.subheader("Métricas no conjunto de teste")
    if output_mode == "classificacao_multiclasse":
        m1, m2 = st.columns(2)
        m1.metric("Acurácia", f"{metricas_teste['acuracia']:.2%}")
        m2.metric("F1 (macro)", f"{metricas_teste['f1_macro']:.3f}")
        st.write("F1 por classe:", {REGIME_CLASSES[i]: round(f, 3) for i, f in enumerate(metricas_teste["f1_por_classe"])})
        st.write("Matriz de confusão (linhas = real, colunas = previsto):")
        st.dataframe(
            pd.DataFrame(metricas_teste["matriz_confusao"], index=REGIME_CLASSES, columns=REGIME_CLASSES),
            use_container_width=True,
        )
        metrica_principal, valor_metrica = "acuracia", metricas_teste["acuracia"]
    elif output_mode == "classificacao_binaria":
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Acurácia", f"{metricas_teste['acuracia']:.2%}")
        m2.metric("Precisão", f"{metricas_teste['precisao']:.2%}")
        m3.metric("Recall", f"{metricas_teste['recall']:.2%}")
        m4.metric("F1", f"{metricas_teste['f1']:.3f}")
        metrica_principal, valor_metrica = "acuracia", metricas_teste["acuracia"]
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metricas_teste['mae']:.6f}")
        m2.metric("RMSE", f"{metricas_teste['rmse']:.6f}")
        m3.metric("R²", f"{metricas_teste['r2']:.4f}")
        metrica_principal, valor_metrica = "mae", metricas_teste["mae"]

    st.subheader("Previsto vs. Real (conjunto de teste)")
    if output_mode == "classificacao_multiclasse":
        df_pred = pd.DataFrame(
            {"real": metricas_teste["reais"], "previsto": metricas_teste["predicoes"]}
        ).reset_index(drop=True)
        st.caption("Códigos de classe: " + ", ".join(f"{i}={n}" for i, n in enumerate(REGIME_CLASSES)))
    else:
        df_pred = pd.DataFrame(
            {"real": metricas_teste["reais"], "previsto": metricas_teste["predicoes"]}
        ).reset_index(drop=True)
    st.line_chart(df_pred, height=280)

    # --- Salvar artefatos localmente e no registry ---
    symbol_nome = nome_dataset.split("_")[0]
    run_id = registry.new_run_id(symbol_nome, arquitetura)
    pasta = registry.run_dir(run_id)

    torch.save(resultado["best_state"], os.path.join(pasta, "model.pt"))
    joblib.dump(scaler, os.path.join(pasta, "scaler.pkl"))
    pd.DataFrame(resultado["loss_history"]).to_json(os.path.join(pasta, "loss_history.json"), orient="records", indent=2)

    metricas_serializaveis = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in metricas_teste.items()
        if k not in ("predicoes", "reais", "probabilidades")
    }
    run_config = {
        "run_id": run_id,
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "dataset": nome_dataset,
        "symbol": symbol_nome,
        "timeframe": nome_dataset.split("_")[1] if "_" in nome_dataset else "",
        "tarefa": tarefa,
        "output_mode": output_mode,
        "output_size": output_size,
        "arquitetura": arquitetura,
        "feature_cols": feature_cols,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": dropout,
        "epochs_treinadas": len(resultado["loss_history"]),
        "melhor_epoca": resultado["best_epoch"],
        "metrica_principal": metrica_principal,
        "valor_metrica": valor_metrica,
        "metricas_teste": metricas_serializaveis,
        "enviado_firebase": False,
    }
    if is_pullback:
        run_config["parametros_pullback"] = {
            "ema_fast": int(ema_fast),
            "ema_slow": int(ema_slow),
            "swing_order": int(swing_order),
            "pullback_horizon": int(pullback_horizon),
            "min_retracement": float(min_retracement),
        }
    if is_regime:
        run_config["parametros_regime"] = {
            "regime_horizon": int(regime_horizon),
            "vol_window": int(vol_window),
            "k_lateral": float(k_lateral),
            "classes": REGIME_CLASSES,
        }

    with open(os.path.join(pasta, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    registry.add_run(run_config)
    st.session_state["ultimo_run_id"] = run_id
    st.info(f"Modelo salvo localmente em `runs/{run_id}/`.")

if "ultimo_run_id" in st.session_state:
    run_id = st.session_state["ultimo_run_id"]
    if firebase_client.is_configured() and st.button("☁️ Enviar este modelo para o Firebase"):
        pasta = registry.run_dir(run_id)
        with st.spinner("Enviando artefatos do modelo ao Firebase..."):
            try:
                firebase_paths = {}
                for arquivo in ["model.pt", "scaler.pkl", "loss_history.json", "config.json"]:
                    caminho_local = os.path.join(pasta, arquivo)
                    remoto = f"{FIREBASE_MODELS_PREFIX}{run_id}/{arquivo}"
                    firebase_client.upload_file(caminho_local, remoto)
                    firebase_paths[arquivo] = remoto
                registry.mark_uploaded(run_id, firebase_paths)
                st.success("Modelo enviado ao Firebase Storage.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Erro ao enviar modelo: {e}")
