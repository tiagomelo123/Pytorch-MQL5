import argparse

import joblib

from .config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    SYMBOL,
    TIMEFRAME,
    SL_PIPS,
    TP_PIPS,
    THRESHOLD,
    model_path,
    processed_data_path,
)
from .evaluate import positive_class_probabilities
from .train import load_processed_dataset


def predict_latest_probability(
    model_file=MODEL_PATH,
    dataset_file=PROCESSED_DATA_PATH,
) -> float:
    if not model_file.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {model_file}. Rode: python -m src.train")

    df = load_processed_dataset(dataset_file)
    if df.empty:
        raise ValueError("Dataset processado esta vazio.")

    model = joblib.load(model_file)
    latest_features = df[FEATURE_COLUMNS].tail(1)
    return float(positive_class_probabilities(model, latest_features)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediz probabilidade do candle mais recente.")
    parser.add_argument("--symbol", default=SYMBOL, help="Ativo. Ex: EURUSD, GBPUSD, USDJPY.")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe. Ex: M1, M5, M15, H1.")
    parser.add_argument("--tp-pips", type=float, default=TP_PIPS, help="Take Profit usado no treino.")
    parser.add_argument("--sl-pips", type=float, default=SL_PIPS, help="Stop Loss usado no treino.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Probabilidade minima.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probability = predict_latest_probability(
        model_path(args.symbol, args.timeframe, args.tp_pips, args.sl_pips),
        processed_data_path(args.symbol, args.timeframe, args.tp_pips, args.sl_pips),
    )
    decision = "ACEITAR" if probability >= args.threshold else "REJEITAR"
    print(f"Probabilidade TP antes do SL: {probability:.4f}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Decisao: {decision}")


if __name__ == "__main__":
    main()
