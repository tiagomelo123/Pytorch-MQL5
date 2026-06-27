import argparse

import joblib

from .backtest import print_backtest_metrics, run_backtest
from .config import (
    SYMBOL,
    TIMEFRAME,
    THRESHOLD,
    SL_PIPS,
    TP_PIPS,
    backtest_report_path,
    learning_curve_path,
    metrics_report_path,
    model_path,
    processed_data_path,
    raw_data_path,
)
from .evaluate import print_metrics
from .features import build_dataset
from .mt5_import import import_from_mt5
from .reporting import save_learning_curve, save_metrics_report
from .train import train_model


def run_pipeline(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    tp_pips: float = TP_PIPS,
    sl_pips: float = SL_PIPS,
    bars: int = 5000,
    threshold: float = THRESHOLD,
    skip_import: bool = False,
) -> None:
    raw_path = raw_data_path(symbol, timeframe)
    dataset_path = processed_data_path(symbol, timeframe, tp_pips, sl_pips)
    output_model_path = model_path(symbol, timeframe, tp_pips, sl_pips)
    output_metrics_path = metrics_report_path(symbol, timeframe, tp_pips, sl_pips)
    output_curve_path = learning_curve_path(symbol, timeframe, tp_pips, sl_pips)
    output_backtest_path = backtest_report_path(symbol, timeframe, tp_pips, sl_pips)

    if not skip_import:
        print(f"Importando {symbol} {timeframe} do MetaTrader 5...")
        imported = import_from_mt5(symbol, timeframe, bars, raw_path)
        print(f"Candles importados: {len(imported)}")

    print("Criando features e labels...")
    dataset = build_dataset(raw_path, dataset_path, symbol, tp_pips, sl_pips)
    print(f"Dataset salvo em: {dataset_path}")
    print(f"TP/SL: {tp_pips:g}/{sl_pips:g} pips")

    print("Treinando rede neural...")
    model, metrics = train_model(dataset)
    metrics["symbol"] = symbol.upper()
    metrics["timeframe"] = timeframe.upper()
    metrics["tp_pips"] = tp_pips
    metrics["sl_pips"] = sl_pips
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model_path)
    save_metrics_report(metrics, output_metrics_path)
    curve_saved = save_learning_curve(model, output_curve_path)
    print_metrics(metrics)
    print(f"Modelo salvo em: {output_model_path}")
    print(f"Metricas salvas em: {output_metrics_path}")
    if curve_saved:
        print(f"Grafico de aprendizagem salvo em: {output_curve_path}")

    print("Rodando backtest...")
    _, backtest_metrics = run_backtest(
        dataset,
        model,
        threshold=threshold,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
    )
    backtest_metrics["symbol"] = symbol.upper()
    backtest_metrics["timeframe"] = timeframe.upper()
    backtest_metrics["tp_pips"] = tp_pips
    backtest_metrics["sl_pips"] = sl_pips
    backtest_metrics["threshold"] = threshold
    save_metrics_report(backtest_metrics, output_backtest_path)
    print_backtest_metrics(backtest_metrics)
    print(f"Metricas do backtest salvas em: {output_backtest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline completo MT5 -> dataset -> treino -> backtest.")
    parser.add_argument("--symbol", default=SYMBOL, help="Ativo. Ex: EURUSD, GBPUSD, USDJPY.")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe. Ex: M1, M5, M15, H1.")
    parser.add_argument("--tp-pips", type=float, default=TP_PIPS, help="Take Profit em pips.")
    parser.add_argument("--sl-pips", type=float, default=SL_PIPS, help="Stop Loss em pips.")
    parser.add_argument("--bars", type=int, default=5000, help="Quantidade de candles do MT5.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Probabilidade minima.")
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Usa o CSV ja existente em data/raw em vez de importar do MT5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        args.symbol,
        args.timeframe,
        args.tp_pips,
        args.sl_pips,
        args.bars,
        args.threshold,
        args.skip_import,
    )


if __name__ == "__main__":
    main()
