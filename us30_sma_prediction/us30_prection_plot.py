import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

H = 10
MA_COL = "ema"          # ou "sma" se você preferir
ATR_COL = "atr"
Y_COL = "y_ema_delta_h10"   # alvo real (do dataset)
YHAT_COL = "yhat"           # previsão do modelo (você cria)

CSV = "out/us30_h1_ma_delta_h10.csv"

# 1) Carrega dataset (precisa ter colunas time + atr + ema/sma + y real + yhat)
df = pd.read_csv(CSV, parse_dates=["time"])

# Se o seu CSV atual não tem ema/atr, você tem 2 opções:
# A) salvar ema/atr também no CSV na etapa de build (recomendado)
# B) recalcular ema/atr aqui do zero (dá, mas prefiro salvar)

# 2) Filtra um período
start = "2024-06-01"
end   = "2024-08-01"
d = df[(df["time"] >= start) & (df["time"] < end)].copy()

# 3) Real: MA no futuro (t+H)
#    Se você já tem a coluna Y_COL, então:
#    (MA_{t+H} - MA_t) = y * ATR_t  => MA_{t+H} = MA_t + y*ATR_t
d["ma_real_t_plus_h"] = d[MA_COL] + d[Y_COL] * d[ATR_COL]

# 4) Previsto: MA no futuro (t+H) com yhat
if YHAT_COL not in d.columns:
    raise RuntimeError(
        "Coluna 'yhat' não encontrada. Gere as previsões e salve em df['yhat'] antes de plotar."
    )

d["ma_pred_t_plus_h"] = d[MA_COL] + d[YHAT_COL] * d[ATR_COL]

# 5) Opcional: comparar deltas (real vs previsto)
d["delta_real_pts"] = d[Y_COL] * d[ATR_COL]
d["delta_pred_pts"] = d[YHAT_COL] * d[ATR_COL]

# 6) Plot 1: MA futura real vs prevista
plt.figure()
plt.plot(d["time"], d["ma_real_t_plus_h"], label=f"{MA_COL.upper()} real t+{H}")
plt.plot(d["time"], d["ma_pred_t_plus_h"], label=f"{MA_COL.upper()} prevista t+{H}")
plt.title(f"MA futura real vs inferida (H={H})")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 7) Plot 2: delta em pontos (real vs previsto)
plt.figure()
plt.plot(d["time"], d["delta_real_pts"], label="Delta real (pts)")
plt.plot(d["time"], d["delta_pred_pts"], label="Delta previsto (pts)")
plt.title(f"Delta da MA em pontos (H={H})")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
