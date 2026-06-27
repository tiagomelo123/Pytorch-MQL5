import argparse

import numpy as np
import pandas as pd

from .config import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SYMBOL,
    TIMEFRAME,
    pip_size_for_symbol,
    processed_data_path,
    raw_data_path,
)
from .labeling import create_buy_labels
from .load_data import load_raw_data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    for window in (1, 3, 5, 10):
        data[f"return_{window}"] = data["close"].pct_change(window)

    for window in (9, 21, 50):
        data[f"ma_{window}"] = data["close"].rolling(window=window).mean()
        data[f"dist_ma_{window}"] = data["close"] - data[f"ma_{window}"]

    data["rsi_14"] = calculate_rsi(data["close"], window=14)
    data["atr_14"] = calculate_atr(data, window=14)
    data["volatility_20"] = data["close"].pct_change().rolling(window=20).std()

    candle_max = data[["open", "close"]].max(axis=1)
    candle_min = data[["open", "close"]].min(axis=1)
    data["body_size"] = (data["close"] - data["open"]).abs()
    data["upper_wick"] = data["high"] - candle_max
    data["lower_wick"] = candle_min - data["low"]
    data["range_size"] = data["high"] - data["low"]

    data["hour"] = data["time"].dt.hour
    data["day_of_week"] = data["time"].dt.dayofweek

    return data


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - prev_close).abs()
    low_close = (df["low"] - prev_close).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def build_dataset(
    raw_path=RAW_DATA_PATH,
    output_path=PROCESSED_DATA_PATH,
    symbol: str = SYMBOL,
) -> pd.DataFrame:
    df = load_raw_data(raw_path)
    dataset = add_features(df)
    dataset["label"] = create_buy_labels(dataset, pip_size=pip_size_for_symbol(symbol))
    dataset = dataset.dropna().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria dataset com features e labels.")
    parser.add_argument("--symbol", default=SYMBOL, help="Ativo. Ex: EURUSD, GBPUSD, USDJPY.")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe. Ex: M1, M5, M15, H1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = raw_data_path(args.symbol, args.timeframe)
    output_path = processed_data_path(args.symbol, args.timeframe)
    dataset = build_dataset(input_path, output_path, args.symbol)
    label_counts = dataset["label"].value_counts().sort_index().to_dict()
    print(f"Dataset salvo em: {output_path}")
    print(f"Linhas: {len(dataset)}")
    print(f"Distribuicao dos labels: {label_counts}")


if __name__ == "__main__":
    main()
