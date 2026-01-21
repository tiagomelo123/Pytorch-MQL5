import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_IN   = BASE_DIR / r"out/us30_h1_ma_delta_h10.csv"
ONNX     = BASE_DIR / r"train_out/us30_h1_cnn1d_ma_delta_h10.onnx"
SCALER   = BASE_DIR / r"train_out/scaler.json"

LOOKBACK = 64
OUT_CSV  = BASE_DIR / r"out/us30_h1_ma_delta_h10_with_yhat.csv"

# Período (opcional):
# - por data: use START/END
# - por índice: USE_INDEX_SLICE=True
START = "2024-06-01"
END   = "2024-08-01"

USE_INDEX_SLICE = False
IDX_START = 20000
IDX_END   = 23000  # exclusivo

# =========================
# HELPERS
# =========================
def load_scaler(path):
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    cols = s["feature_cols"]
    mean = np.array(s["mean"], dtype=np.float32)
    std  = np.array(s["std"], dtype=np.float32)
    eps  = float(s.get("eps", 1e-8))
    return cols, mean, std, eps

def make_windows(X2d, lookback):
    """
    X2d: (T, F)
    retorna:
      Xw: (T-(lookback-1), F, lookback)
      start_idx = lookback-1 (primeiro t que tem janela completa)
    """
    T, F = X2d.shape
    start_idx = lookback - 1
    n = T - start_idx
    if n <= 0:
        raise ValueError("Poucos dados para o lookback escolhido.")
    Xw = np.zeros((n, F, lookback), dtype=np.float32)
    for i in range(n):
        t = start_idx + i
        w = X2d[t - lookback + 1 : t + 1]  # (lookback, F)
        Xw[i] = w.T                        # (F, lookback)
    return Xw, start_idx

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_IN)
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

# recorte opcional
if USE_INDEX_SLICE:
    d = df.iloc[IDX_START:IDX_END].copy()
else:
    if "time" not in df.columns:
        raise RuntimeError("Para recorte por data, o CSV precisa ter coluna 'time'.")
    d = df[(df["time"] >= START) & (df["time"] < END)].copy()

d = d.reset_index(drop=True)

# sanity
if len(d) < LOOKBACK + 5:
    raise RuntimeError(
        f"Período pequeno: {len(d)} linhas. "
        f"Use pelo menos {LOOKBACK + 5} (ideal 400+)."
    )

# =========================
# SCALER + FEATURES
# =========================
feature_cols, mean, std, eps = load_scaler(SCALER)

missing = [c for c in feature_cols if c not in d.columns]
if missing:
    raise RuntimeError(f"Faltam features no CSV: {missing}")

X = d[feature_cols].astype(np.float32).to_numpy()
Xs = (X - mean) / (std + eps)

# =========================
# WINDOW + ONNX INFER
# =========================
Xw, start_idx = make_windows(Xs, LOOKBACK)

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

yhat = np.full((len(d),), np.nan, dtype=np.float32)

# infer batch=1 (simples e seguro)
for i in range(Xw.shape[0]):
    pred = sess.run([out_name], {inp_name: Xw[i:i+1]})[0]  # (1,1)
    yhat[start_idx + i] = float(pred[0, 0])

d["yhat"] = yhat

# =========================
# SAVE
# =========================
d.to_csv(OUT_CSV, index=False)
print(f"OK: inferência concluída e CSV salvo em: {OUT_CSV}")
print(f"Linhas no recorte: {len(d)} | yhat válidos: {np.isfinite(d['yhat']).sum()}")