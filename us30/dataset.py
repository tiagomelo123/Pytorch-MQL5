import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# =========================
# Config
# =========================
SYMBOL = "US30"          # ajuste para seu símbolo (ex: "US30", "US30.cash", etc.)
TIMEFRAME = mt5.TIMEFRAME_D1
N_BARS = 100           # quantos candles puxar

ATR_PERIOD = 14
HORIZON = 5
TP_ATR = 1.5
SL_ATR = 1.1

BASE_DIR = Path(__file__).resolve().parent
OUT_CSV = BASE_DIR / "dataset.csv"


# =========================
# Utils
# =========================
def ensure_mt5_initialized():
    if mt5.initialize():
        return True

    # Se falhar, mostre erro e saia
    err = mt5.last_error()
    raise RuntimeError(f"MT5 initialize() falhou: {err}")


def fetch_rates(symbol: str, timeframe, n_bars: int) -> pd.DataFrame:
    # Garante que o símbolo está visível
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Símbolo não encontrado no MT5: {symbol}")

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Não foi possível selecionar símbolo: {symbol}")

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates retornou vazio para {symbol}")

    df = pd.DataFrame(rates)
    # time vem em segundos (epoch)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    # padroniza nomes
    df.rename(columns={"tick_volume": "tick_volume"}, inplace=True)

    return df

def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(period).mean().to_numpy()
    return atr


def create_labels_3class_barrier(df: pd.DataFrame,
                                horizon: int = 10,
                                atr_period: int = 14,
                                tp_atr: float = 1.0,
                                sl_atr: float = 0.7) -> pd.DataFrame:
    """
    labels:
      0 = NEUTRAL
      1 = UP
      2 = RISK
    """
    df = df.copy()
    df["atr"] = compute_atr(df, period=atr_period)

    n = len(df)
    labels = np.zeros(n, dtype=np.int64)

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()

    for i in range(n):
        if np.isnan(atr[i]):
            labels[i] = 0
            continue

        tp = close[i] + tp_atr * atr[i]
        sl = close[i] - sl_atr * atr[i]

        end = min(i + horizon, n - 1)
        outcome = 0  # neutral

        for j in range(i + 1, end + 1):
            hit_tp = high[j] >= tp
            hit_sl = low[j] <= sl

            if hit_tp and hit_sl:
                # empate no mesmo candle -> conservador
                outcome = 2
                break
            if hit_tp:
                outcome = 1
                break
            if hit_sl:
                outcome = 2
                break

        labels[i] = outcome

    df["label"] = labels
    return df


def main():
    ensure_mt5_initialized()

    try:
        df = fetch_rates(SYMBOL, TIMEFRAME, N_BARS)

        # Se quiser incluir spread/real volume:
        # df já vem com 'spread' e 'real_volume' dependendo do ativo/corretora.

        df2 = create_labels_3class_barrier(
            df,
            horizon=HORIZON,
            atr_period=ATR_PERIOD,
            tp_atr=TP_ATR,
            sl_atr=SL_ATR
        )

        df2["label_name"] = df2["label"].map({0: "NEUTRAL", 1: "UP", 2: "RISK"})

        df2.to_csv(OUT_CSV, index=False)

        counts = df2["label_name"].value_counts(dropna=False).to_dict()
        print("OK ✅ CSV gerado:", OUT_CSV)
        print("Distribuição:", counts)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
