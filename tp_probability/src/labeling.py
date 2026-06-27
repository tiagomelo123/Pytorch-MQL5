import numpy as np
import pandas as pd

from .config import MAX_BARS_AHEAD, PIP_SIZE, SL_PIPS, TP_PIPS


def create_buy_labels(
    df: pd.DataFrame,
    tp_pips: float = TP_PIPS,
    sl_pips: float = SL_PIPS,
    pip_size: float = PIP_SIZE,
    max_bars_ahead: int = MAX_BARS_AHEAD,
) -> pd.Series:
    labels = np.full(len(df), -1, dtype=int)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    for index in range(len(df)):
        entry = closes[index]
        tp_price = entry + tp_pips * pip_size
        sl_price = entry - sl_pips * pip_size
        end = min(index + max_bars_ahead, len(df) - 1)

        for future_index in range(index + 1, end + 1):
            hit_tp = highs[future_index] >= tp_price
            hit_sl = lows[future_index] <= sl_price

            if hit_sl:
                labels[index] = 0
                break
            if hit_tp:
                labels[index] = 1
                break

    return pd.Series(labels, index=df.index, name="label")
