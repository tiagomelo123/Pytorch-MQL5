import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# =========================
# CONFIG
# =========================
SYMBOL     = "US30"
TIMEFRAME  = mt5.TIMEFRAME_H1
N_BARS     = 50000
HORIZON_H  = 20
MA_PERIOD  = 50
ATR_PERIOD = 14

OUT_CSV = BASE_DIR / "out/us30_h1_ma_delta_h10.csv"

# =========================
# INIT MT5
# =========================
if not mt5.initialize():
    raise RuntimeError("Erro ao inicializar MetaTrader5")

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, N_BARS)
mt5.shutdown()

if rates is None or len(rates) == 0:
    raise RuntimeError("Nenhum dado retornado do MT5")

# =========================
# DATAFRAME
# =========================
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")

# =========================
# INDICADORES
# =========================
df["ema"] = EMAIndicator(df["close"], window=MA_PERIOD).ema_indicator()
df["sma"] = SMAIndicator(df["close"], window=MA_PERIOD).sma_indicator()

atr = AverageTrueRange(
    high=df["high"],
    low=df["low"],
    close=df["close"],
    window=ATR_PERIOD
)
df["atr"] = atr.average_true_range()

# =========================
# FEATURES (somente passado)
# =========================
df["ret_1"] = np.log(df["close"] / df["close"].shift(1))
df["hl"] = (df["high"] - df["low"]) / df["atr"]
df["body"] = (df["close"] - df["open"]) / df["atr"]
df["dist_ema"] = (df["close"] - df["ema"]) / df["atr"]
df["slope_ema"] = (df["ema"] - df["ema"].shift(1)) / df["atr"]
df["atr_norm"] = df["atr"] / df["close"]

# =========================
# LABEL — DELTA FUTURO DA MA
# y = (EMA[t+H] - EMA[t]) / ATR[t]
# =========================
df["y_ema_delta_h10"] = (
    df["ema"].shift(-HORIZON_H) - df["ema"]
) / df["atr"]

df["y_sma_delta_h10"] = (
    df["sma"].shift(-HORIZON_H) - df["sma"]
) / df["atr"]

# =========================
# LIMPEZA
# =========================
df = df.dropna().reset_index(drop=True)

# =========================
# COLUNAS FINAIS
# =========================
OHLC_COLS = ["open", "high", "low", "close"]

FEATURE_COLS = [
    "ret_1",
    "hl",
    "body",
    "dist_ema",
    "slope_ema",
    "atr_norm"
]

TARGET_COLS = [
    "y_ema_delta_h10",
    "y_sma_delta_h10"
]

FINAL_COLS = ["time"] + OHLC_COLS + FEATURE_COLS + TARGET_COLS

df_out = df[FINAL_COLS]

# =========================
# SAVE CSV
# =========================
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_out.to_csv(OUT_CSV, index=False)
print(f"OK: dataset salvo em {OUT_CSV}")
print("Rows:", len(df_out))