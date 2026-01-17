import json
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

# reaproveita suas funções do dataset_macd.py
from dataset_macd import build_features, mt5_connect, find_us30_symbol, ensure_symbol_selected, fetch_rates
import MetaTrader5 as mt5

BASE_DIR = Path(__file__).resolve().parent

def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def load_scalper_json(path: str):
    """
    Tentativa de ler scalper.json de forma tolerante:
    - procura por feature_cols (ou feature_columns)
    - procura mean/std (ou scaler.mean/scaler.std etc.)
    - procura prob_class_idx (qual coluna é "CONT")
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    def pick(*keys, default=None):
        cur = cfg
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    feature_cols = cfg.get("feature_cols") or cfg.get("feature_columns")
    if feature_cols is None:
        # fallback: tenta caminhos comuns
        feature_cols = pick("scaler", "feature_cols") or pick("scaler", "feature_columns")

    mean = cfg.get("mean")
    std = cfg.get("std")
    if mean is None or std is None:
        mean = pick("scaler", "mean")
        std  = pick("scaler", "std")

    prob_class_idx = cfg.get("prob_class_idx", 1)  # default 1 (ajuste se necessário)
    return feature_cols, mean, std, prob_class_idx

def run_onnx_probs(model_path: str, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape  # ex: [1, 24]

    # Se o modelo exige batch=1, roda amostra por amostra
    if isinstance(in_shape, list) and len(in_shape) >= 1 and in_shape[0] == 1 and X.shape[0] != 1:
        outs = []
        for i in range(X.shape[0]):
            y = sess.run(None, {in_name: X[i:i+1].astype(np.float32)})[0]
            outs.append(y)
        y = np.concatenate(outs, axis=0)
    else:
        y = sess.run(None, {in_name: X.astype(np.float32)})[0]

    # logits (N,C) -> softmax
    if y.ndim == 2 and y.shape[1] >= 2:
        return softmax(y, axis=1)

    # prob (N,) -> (N,1)
    if y.ndim == 1:
        return y.reshape(-1, 1)

    return y


def main():
    MODEL_PATH =  BASE_DIR / "train_out/us30_macd_mlp.onnx"
    SCALPER_JSON = BASE_DIR / "train_out/scaler.json"

    OUT_DIR = BASE_DIR / "out"
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_CSV = os.path.join(OUT_DIR, "us30_h1_nn_probs.csv")

    N_BARS = 10000
    TIMEFRAME = mt5.TIMEFRAME_H1
    FORCE_SYMBOL = None  # ex: "US30.cash"

    # ---- load config (scaler/feature order) ----
    feature_cols, mean, std, prob_class_idx = load_scalper_json(SCALPER_JSON)

    # Se seu scalper.json NÃO tiver feature_cols, use a mesma lista do dataset_macd.py :contentReference[oaicite:1]{index=1}
    if feature_cols is None:
        feature_cols = (
            [f"ret_{k}" for k in range(1, 8)]
            + ["ret_sum_3", "ret_vol_7"]
            + ["macd_hist", "macd_hist_slope_1", "macd_hist_slope_3"]
            + ["ema50_slope_5", "dist_close_ema50"]
            + ["adx14", "atr_pct", "bb_width_20", "macd_compress"]
        )

    mean = np.array(mean, dtype=np.float32) if mean is not None else None
    std  = np.array(std, dtype=np.float32) if std  is not None else None

    # ---- MT5 fetch ----
    mt5_connect()  # assume MT5 aberto/logado
    symbol = find_us30_symbol(prefer=FORCE_SYMBOL)
    ensure_symbol_selected(symbol)

    df_rates = fetch_rates(symbol, TIMEFRAME, N_BARS)
    df_ohlc = df_rates[["open", "high", "low", "close"]].copy()

    # ---- features ----
    feat = build_features(df_ohlc)

    # precisamos manter time/close e garantir mesmas features sem NaN
    d = feat[["close"] + list(feature_cols)].dropna().copy()

    X = d[feature_cols].astype(np.float32).to_numpy()

    # ---- normalize like training ----
    if mean is not None and std is not None:
        if mean.shape[0] != X.shape[1] or std.shape[0] != X.shape[1]:
            raise ValueError(f"mean/std não batem com #features: mean={mean.shape}, std={std.shape}, X={X.shape}")
        std_safe = np.where(std == 0, 1.0, std)
        X = (X - mean) / std_safe

    # ---- infer ----
    probs = run_onnx_probs(MODEL_PATH, X)

    # coluna da classe CONT
    if probs.ndim == 2 and probs.shape[1] >= 2:
        d["p_cont"] = probs[:, prob_class_idx]
        # opcional: guardar todas
        # for i in range(probs.shape[1]):
        #     d[f"p_{i}"] = probs[:, i]
    else:
        d["p_cont"] = probs[:, 0]

    # time está no index (porque fetch_rates set_index("time"))
    out = d.reset_index().rename(columns={"index": "time"})
    out = out[["time", "close", "p_cont"]]

    out.to_csv(OUT_CSV, index=False)
    print("OK:", OUT_CSV, "rows=", len(out))

    mt5.shutdown()

if __name__ == "__main__":
    main()
