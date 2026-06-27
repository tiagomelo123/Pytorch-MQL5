import argparse

import joblib
import numpy as np
import pandas as pd

from .config import (
    COMMISSION_PIPS,
    FEATURE_COLUMNS,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    SYMBOL,
    SL_PIPS,
    SLIPPAGE_PIPS,
    SPREAD_PIPS,
    TIMEFRAME,
    THRESHOLD,
    TP_PIPS,
    TRAIN_SIZE,
    backtest_report_path,
    model_path,
    processed_data_path,
)
from .evaluate import positive_class_probabilities
from .reporting import save_metrics_report
from .train import load_processed_dataset, temporal_train_test_split


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {path}. Rode: python -m src.train")
    return joblib.load(path)


def run_backtest(
    df: pd.DataFrame,
    model,
    threshold: float = THRESHOLD,
    tp_pips: float = TP_PIPS,
    sl_pips: float = SL_PIPS,
    spread_pips: float = SPREAD_PIPS,
    slippage_pips: float = SLIPPAGE_PIPS,
    commission_pips: float = COMMISSION_PIPS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    usable = df[df["label"] != -1].copy()
    _, test_df = temporal_train_test_split(usable, TRAIN_SIZE)
    test_df = test_df.copy()
    test_df["prob_tp"] = positive_class_probabilities(model, test_df[FEATURE_COLUMNS])
    trades = test_df[test_df["prob_tp"] >= threshold].copy()

    total_cost = spread_pips + slippage_pips + commission_pips
    trades["pips"] = np.where(
        trades["label"].astype(int) == 1,
        tp_pips - total_cost,
        -sl_pips - total_cost,
    )
    trades["equity"] = trades["pips"].cumsum()

    metrics = calculate_backtest_metrics(trades)
    return trades, metrics


def calculate_backtest_metrics(trades: pd.DataFrame) -> dict[str, float]:
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "max_losing_streak": 0,
            "avg_pips_per_trade": 0.0,
        }

    wins = trades[trades["pips"] > 0]
    losses = trades[trades["pips"] < 0]
    gross_profit = wins["pips"].sum()
    gross_loss = losses["pips"].abs().sum()
    equity = trades["equity"]
    drawdown = equity.cummax() - equity

    return {
        "total_trades": total_trades,
        "win_rate": len(wins) / total_trades,
        "net_pips": trades["pips"].sum(),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_drawdown": drawdown.max(),
        "expectancy": trades["pips"].mean(),
        "max_losing_streak": max_losing_streak(trades["pips"]),
        "avg_pips_per_trade": trades["pips"].mean(),
    }


def max_losing_streak(pips: pd.Series) -> int:
    current = 0
    longest = 0
    for value in pips:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def print_backtest_metrics(metrics: dict[str, float]) -> None:
    print("Metricas do backtest")
    print(f"Total de trades: {metrics['total_trades']}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Lucro/prejuizo em pips: {metrics['net_pips']:.2f}")
    print(f"Profit factor: {metrics['profit_factor']:.4f}")
    print(f"Drawdown maximo: {metrics['max_drawdown']:.2f}")
    print(f"Expectancy por trade: {metrics['expectancy']:.2f}")
    print(f"Sequencia maxima de perdas: {metrics['max_losing_streak']}")
    print(f"Media de pips por trade: {metrics['avg_pips_per_trade']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda backtest com threshold.")
    parser.add_argument("--symbol", default=SYMBOL, help="Ativo. Ex: EURUSD, GBPUSD, USDJPY.")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe. Ex: M1, M5, M15, H1.")
    parser.add_argument("--tp-pips", type=float, default=TP_PIPS, help="Take Profit usado no treino.")
    parser.add_argument("--sl-pips", type=float, default=SL_PIPS, help="Stop Loss usado no treino.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Probabilidade minima.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(model_path(args.symbol, args.timeframe, args.tp_pips, args.sl_pips))
    df = load_processed_dataset(processed_data_path(args.symbol, args.timeframe, args.tp_pips, args.sl_pips))
    _, metrics = run_backtest(
        df,
        model,
        threshold=args.threshold,
        tp_pips=args.tp_pips,
        sl_pips=args.sl_pips,
    )
    metrics["symbol"] = args.symbol.upper()
    metrics["timeframe"] = args.timeframe.upper()
    metrics["tp_pips"] = args.tp_pips
    metrics["sl_pips"] = args.sl_pips
    metrics["threshold"] = args.threshold
    output_metrics_path = backtest_report_path(args.symbol, args.timeframe, args.tp_pips, args.sl_pips)
    save_metrics_report(metrics, output_metrics_path)
    print_backtest_metrics(metrics)
    print(f"Metricas do backtest salvas em: {output_metrics_path}")


if __name__ == "__main__":
    main()
