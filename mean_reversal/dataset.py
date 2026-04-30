# mean_reversion_dataset.py

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# CONFIGURAÇÕES
# =========================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
START_DATE = datetime(2025, 1, 1)
END_DATE   = datetime(2026, 4, 28)

BB_PERIOD = 20
BB_DEV = 2.5

ATR_PERIOD = 14
ADX_PERIOD = 14
RSI_PERIOD = 14

PIVOT_BUFFER_ATR = 0.30

# =========================
# CONFIGURAÇÕES NOVAS
# =========================

TP_ATR_MULT = 1.0
SL_ATR_MULT = 0.8
SL_STRUCTURE_BUFFER_ATR = 0.20
LOOKAHEAD_BARS = 20

OUTPUT_FILE = "mean_reversion_dataset.csv"


# =========================
# INDICADORES
# =========================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain, index=series.index).rolling(period).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def adx(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr = atr(df, period)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


# =========================
# PIVOT DIÁRIO
# =========================

def add_daily_pivots(df):
    df["date"] = df["time"].dt.date

    daily = df.groupby("date").agg({
        "high": "max",
        "low": "min",
        "close": "last"
    })

    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_close"] = daily["close"].shift(1)

    daily["pivot"] = (daily["prev_high"] + daily["prev_low"] + daily["prev_close"]) / 3
    daily["r1"] = (2 * daily["pivot"]) - daily["prev_low"]
    daily["s1"] = (2 * daily["pivot"]) - daily["prev_high"]
    daily["r2"] = daily["pivot"] + (daily["prev_high"] - daily["prev_low"])
    daily["s2"] = daily["pivot"] - (daily["prev_high"] - daily["prev_low"])

    df = df.merge(
        daily[["pivot", "r1", "s1", "r2", "s2"]],
        left_on="date",
        right_index=True,
        how="left"
    )

    return df


# =========================
# FRACTAIS
# =========================

def add_fractals(df):
    df["fractal_high"] = (
        (df["high"].shift(2) < df["high"]) &
        (df["high"].shift(1) < df["high"]) &
        (df["high"].shift(-1) < df["high"]) &
        (df["high"].shift(-2) < df["high"])
    )

    df["fractal_low"] = (
        (df["low"].shift(2) > df["low"]) &
        (df["low"].shift(1) > df["low"]) &
        (df["low"].shift(-1) > df["low"]) &
        (df["low"].shift(-2) > df["low"])
    )

    df["last_fractal_high"] = df["high"].where(df["fractal_high"]).ffill()
    df["last_fractal_low"] = df["low"].where(df["fractal_low"]).ffill()

    return df


# =========================
# FEATURES
# =========================

def create_features(df):
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["sma20"] = sma(df["close"], BB_PERIOD)

    df["std20"] = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = df["sma20"] + BB_DEV * df["std20"]
    df["bb_lower"] = df["sma20"] - BB_DEV * df["std20"]

    df["atr14"] = atr(df, ATR_PERIOD)
    df["adx14"] = adx(df, ADX_PERIOD)
    df["rsi14"] = rsi(df["close"], RSI_PERIOD)

    df["zscore"] = (df["close"] - df["sma20"]) / df["std20"]
    df["dist_ema20_atr"] = (df["close"] - df["ema20"]) / df["atr14"]
    df["dist_ema50_atr"] = (df["close"] - df["ema50"]) / df["atr14"]

    df["bb_width_atr"] = (df["bb_upper"] - df["bb_lower"]) / df["atr14"]
    df["bb_percent_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["touch_bb_upper"] = (df["high"] >= df["bb_upper"]).astype(int)
    df["touch_bb_lower"] = (df["low"] <= df["bb_lower"]).astype(int)

    df["close_outside_upper"] = (df["close"] > df["bb_upper"]).astype(int)
    df["close_outside_lower"] = (df["close"] < df["bb_lower"]).astype(int)

    df["range_atr"] = (df["high"] - df["low"]) / df["atr14"]
    df["body_atr"] = (df["close"] - df["open"]).abs() / df["atr14"]

    candle_range = df["high"] - df["low"]
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["close_position"] = (df["close"] - df["low"]) / candle_range

    df["ema20_slope"] = (df["ema20"] - df["ema20"].shift(5)) / df["atr14"]
    df["ema50_slope"] = (df["ema50"] - df["ema50"].shift(5)) / df["atr14"]
    df["ema_distance_atr"] = (df["ema20"] - df["ema50"]) / df["atr14"]

    df["roc_1"] = df["close"].pct_change(1)
    df["roc_3"] = df["close"].pct_change(3)
    df["roc_5"] = df["close"].pct_change(5)

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek

   

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df = add_daily_pivots(df)
    df = add_fractals(df)

    for level in ["pivot", "r1", "r2", "s1", "s2"]:
        df[f"dist_{level}_atr"] = (df["close"] - df[level]) / df["atr14"]

    df["near_r1"] = (df["dist_r1_atr"].abs() <= PIVOT_BUFFER_ATR).astype(int)
    df["near_r2"] = (df["dist_r2_atr"].abs() <= PIVOT_BUFFER_ATR).astype(int)
    df["near_s1"] = (df["dist_s1_atr"].abs() <= PIVOT_BUFFER_ATR).astype(int)
    df["near_s2"] = (df["dist_s2_atr"].abs() <= PIVOT_BUFFER_ATR).astype(int)

    df["dist_fractal_high_atr"] = (df["close"] - df["last_fractal_high"]) / df["atr14"]
    df["dist_fractal_low_atr"] = (df["close"] - df["last_fractal_low"]) / df["atr14"]

    return df

def add_directional_features(df):
    df["trend_strength"] = df["adx14"] * df["range_atr"]
    df["impulse_strength"] = df["body_atr"] * df["range_atr"]

    df["ema_slope_agreement"] = (
        np.sign(df["ema20_slope"]) * np.sign(df["ema50_slope"])
    )

    df["bb_touch_direction"] = (
        ((df["direction"] == -1) & (df["touch_bb_upper"] == 1)) |
        ((df["direction"] == 1) & (df["touch_bb_lower"] == 1))
    ).astype(int)

    df["outside_bb_direction"] = (
        ((df["direction"] == -1) & (df["close_outside_upper"] == 1)) |
        ((df["direction"] == 1) & (df["close_outside_lower"] == 1))
    ).astype(int)

    df["wick_reversal_direction"] = np.where(
        df["direction"] == 1,
        df["lower_wick_ratio"],
        df["upper_wick_ratio"]
    )

    df["rsi_reversal_strength"] = np.where(
        df["direction"] == 1,
        np.maximum(30 - df["rsi14"], 0),
        np.maximum(df["rsi14"] - 70, 0)
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Confluência com pivôs na direção da reversão
    df["near_resistance_direction"] = (
        (df["direction"] == -1) &
        ((df["near_r1"] == 1) | (df["near_r2"] == 1))
    ).astype(int)

    df["near_support_direction"] = (
        (df["direction"] == 1) &
        ((df["near_s1"] == 1) | (df["near_s2"] == 1))
    ).astype(int)

    # Distância ao nível de pivot mais favorável para a reversão
    df["dist_reversal_pivot_atr"] = np.where(
        df["direction"] == -1,
        np.minimum(df["dist_r1_atr"].abs(), df["dist_r2_atr"].abs()),
        np.minimum(df["dist_s1_atr"].abs(), df["dist_s2_atr"].abs())
    )

    # Fractal favorável na direção da reversão
    df["dist_reversal_fractal_atr"] = np.where(
        df["direction"] == -1,
        df["dist_fractal_high_atr"].abs(),
        df["dist_fractal_low_atr"].abs()
    )

    # Distância da entrada até o stop/target em pontos de preço
    df["tp_distance_price"] = abs(df["tp_price"] - df["close"])
    df["sl_distance_price"] = abs(df["sl_price"] - df["close"])

    return df

# =========================
# LABELS
# =========================

def create_labels(df):
    labels = []
    directions = []
    tp_prices = []
    sl_prices = []

    for i in range(len(df)):
        row = df.iloc[i]

        label = np.nan
        direction = 0
        tp = np.nan
        sl = np.nan

        if (
            np.isnan(row["atr14"]) or
            np.isnan(row["bb_upper"]) or
            np.isnan(row["bb_lower"])
        ):
            labels.append(label)
            directions.append(direction)
            tp_prices.append(tp)
            sl_prices.append(sl)
            continue

        entry = row["close"]
        atr_value = row["atr14"]
        future = df.iloc[i + 1:i + 1 + LOOKAHEAD_BARS]

        if len(future) < LOOKAHEAD_BARS:
            labels.append(label)
            directions.append(direction)
            tp_prices.append(tp)
            sl_prices.append(sl)
            continue

        # =========================
        # VENDA: preço esticado para cima
        # =========================
        if row["close"] > row["bb_upper"]:
            direction = -1

            # TP por ATR
            tp = entry - (TP_ATR_MULT * atr_value)

            # SL estrutural:
            # usa fractal high se existir; caso contrário usa SL ATR
            sl_atr = entry + (SL_ATR_MULT * atr_value)

            if not np.isnan(row.get("last_fractal_high", np.nan)):
                sl_structure = row["last_fractal_high"] + (SL_STRUCTURE_BUFFER_ATR * atr_value)
                sl = max(sl_atr, sl_structure)
            else:
                sl = sl_atr

            for _, f in future.iterrows():
                if f["high"] >= sl:
                    label = 0
                    break
                if f["low"] <= tp:
                    label = 1
                    break

        # =========================
        # COMPRA: preço esticado para baixo
        # =========================
        elif row["close"] < row["bb_lower"]:
            direction = 1

            # TP por ATR
            tp = entry + (TP_ATR_MULT * atr_value)

            # SL estrutural:
            # usa fractal low se existir; caso contrário usa SL ATR
            sl_atr = entry - (SL_ATR_MULT * atr_value)

            if not np.isnan(row.get("last_fractal_low", np.nan)):
                sl_structure = row["last_fractal_low"] - (SL_STRUCTURE_BUFFER_ATR * atr_value)
                sl = min(sl_atr, sl_structure)
            else:
                sl = sl_atr

            for _, f in future.iterrows():
                if f["low"] <= sl:
                    label = 0
                    break
                if f["high"] >= tp:
                    label = 1
                    break

        labels.append(label)
        directions.append(direction)
        tp_prices.append(tp)
        sl_prices.append(sl)

    df["direction"] = directions
    df["tp_price"] = tp_prices
    df["sl_price"] = sl_prices
    df["label"] = labels

    df["tp_dist_atr"] = abs(df["tp_price"] - df["close"]) / df["atr14"]
    df["sl_dist_atr"] = abs(df["sl_price"] - df["close"]) / df["atr14"]
    df["rr"] = df["tp_dist_atr"] / df["sl_dist_atr"]

    return df


# =========================
# MT5
# =========================

def get_mt5_data(symbol, timeframe, start_date, end_date):
    if not mt5.initialize():
        raise RuntimeError("Erro ao inicializar MetaTrader 5")

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Não foi possível selecionar o ativo {symbol}")

    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("Nenhum dado retornado pelo MetaTrader 5")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df


# =========================
# MAIN
# =========================

def main():
    df = get_mt5_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)

    df = create_features(df)
    df = create_labels(df)

    df = df.dropna()

    # Mantém somente setups reais
    df = df[df["direction"] != 0]

    df = add_directional_features(df)

    feature_cols = [
        "open", "high", "low", "close", "tick_volume",
        "ema20", "ema50",
        "atr14", "adx14", "rsi14",
        "zscore",
        "dist_ema20_atr",
        "dist_ema50_atr",
        "bb_width_atr",
        "bb_percent_b",
        "touch_bb_upper",
        "touch_bb_lower",
        "close_outside_upper",
        "close_outside_lower",
        "range_atr",
        "body_atr",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position",
        "ema20_slope",
        "ema50_slope",
        "ema_distance_atr",
        "roc_1",
        "roc_3",
        "roc_5",
        "hour",
        "day_of_week",
        "dist_pivot_atr",
        "dist_r1_atr",
        "dist_r2_atr",
        "dist_s1_atr",
        "dist_s2_atr",
        "near_r1",
        "near_r2",
        "near_s1",
        "near_s2",
        "dist_fractal_high_atr",
        "dist_fractal_low_atr",
        "direction",
        "label",
        "trend_strength",
        "impulse_strength",
        "ema_slope_agreement",
        "bb_touch_direction",
        "outside_bb_direction",
        "wick_reversal_direction",
        "rsi_reversal_strength",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "near_resistance_direction",
        "near_support_direction",
        "dist_reversal_pivot_atr",
        "dist_reversal_fractal_atr",
        "tp_distance_price",
        "sl_distance_price",
        "tp_dist_atr",
        "sl_dist_atr",
        "rr",
    ]

    df_final = df[["time"] + feature_cols]

    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset salvo em: {OUTPUT_FILE}")
    print(df_final["label"].value_counts())
    print(df_final.head())


if __name__ == "__main__":
    main()