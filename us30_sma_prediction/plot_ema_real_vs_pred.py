import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from pathlib import Path


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_IN = BASE_DIR / r"out/us30_h1_ma_delta_h10_with_yhat.csv"

H = 20
EMA_PERIOD = 50
ATR_PERIOD = 14

# =========================
# SAVE FIG CONFIG
# =========================
SAVE_FIG = True
PLOTS_DIR = BASE_DIR / Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

FIG_NAME = f"ema_real_vs_pred_h{H}_ema{EMA_PERIOD}_lb64.png"
FIG_PATH =   PLOTS_DIR / FIG_NAME

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_IN)
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

required = {"open","high","low","close","yhat"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Faltam colunas no CSV para plot: {missing}")

# =========================
# RECALC EMA/ATR
# =========================
df["ema"] = EMAIndicator(df["close"], window=EMA_PERIOD).ema_indicator()
atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_PERIOD)
df["atr"] = atr.average_true_range()

# =========================
# BUILD SERIES
# yhat = (EMA[t+H] - EMA[t]) / ATR[t]
# => EMA_pred(t+H) = EMA[t] + yhat*ATR[t]
# EMA_real(t+H) = EMA[t+H]
# =========================
df["ema_real_t_plus_h"] = df["ema"].shift(-H)
df["ema_pred_t_plus_h"] = df["ema"] + df["yhat"] * df["atr"]
df["time_future"] = df["time"].shift(-H)

plot_df = df.dropna(subset=["yhat", "ema_real_t_plus_h", "ema_pred_t_plus_h", "time_future"]).copy()

print(f"Plot rows: {len(plot_df)} (de {len(df)})")

# =========================
# PLOT
# =========================
plt.figure()
plt.plot(plot_df["time_future"], plot_df["ema_real_t_plus_h"], label=f"EMA real (t+{H})")
plt.plot(plot_df["time_future"], plot_df["ema_pred_t_plus_h"], label=f"EMA prevista (t+{H})")
plt.title(f"EMA real vs EMA prevista | H={H}, EMA={EMA_PERIOD}, ATR={ATR_PERIOD}")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(FIG_PATH, dpi=150)
    print(f"OK: figura salva em {FIG_PATH}")

plt.show()  # opcional, pode comentar se quiser só salvar