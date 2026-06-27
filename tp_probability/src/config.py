from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

SYMBOL = "EURUSD"
TIMEFRAME = "M5"

TP_PIPS = 20
SL_PIPS = 15
PIP_SIZE = 0.0001
MAX_BARS_AHEAD = 50

TRAIN_SIZE = 0.80
THRESHOLD = 0.70

SPREAD_PIPS = 1.0
SLIPPAGE_PIPS = 0.2
COMMISSION_PIPS = 0.0

REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "ma_9",
    "ma_21",
    "ma_50",
    "dist_ma_9",
    "dist_ma_21",
    "dist_ma_50",
    "rsi_14",
    "atr_14",
    "volatility_20",
    "body_size",
    "upper_wick",
    "lower_wick",
    "range_size",
    "hour",
    "day_of_week",
]


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def normalize_timeframe(timeframe: str) -> str:
    return timeframe.strip().upper()


def dataset_key(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> str:
    return f"{normalize_symbol(symbol).lower()}_{normalize_timeframe(timeframe).lower()}"


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if normalize_symbol(symbol).endswith("JPY") else 0.0001


def raw_data_path(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Path:
    return BASE_DIR / "data" / "raw" / f"{dataset_key(symbol, timeframe)}.csv"


def processed_data_path(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Path:
    return BASE_DIR / "data" / "processed" / f"dataset_{dataset_key(symbol, timeframe)}.csv"


def model_path(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Path:
    return BASE_DIR / "models" / f"tp_sl_classifier_{dataset_key(symbol, timeframe)}.pkl"


def report_dir(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Path:
    return BASE_DIR / "reports" / dataset_key(symbol, timeframe)


def metrics_report_path(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Path:
    return report_dir(symbol, timeframe) / "metrics.json"


def learning_curve_path(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Path:
    return report_dir(symbol, timeframe) / "learning_curve.png"


def backtest_report_path(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Path:
    return report_dir(symbol, timeframe) / "backtest_metrics.json"


RAW_DATA_PATH = raw_data_path(SYMBOL, TIMEFRAME)
PROCESSED_DATA_PATH = processed_data_path(SYMBOL, TIMEFRAME)
MODEL_PATH = model_path(SYMBOL, TIMEFRAME)
