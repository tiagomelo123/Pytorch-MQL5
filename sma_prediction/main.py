"""Entry point CLI do pipeline de previsão de média móvel.

Sem lógica de negócio aqui: apenas parse de argumentos e despacho para o
``pipeline.runner`` (modo train) ou ``predict.inference`` (modo predict).
"""

import argparse
import json
import logging
import sys

from config import build_config
from pipeline.runner import build_run_dir, run_pipeline
from predict.inference import predict_next


def _setup_logging() -> None:
    """Configura logging em nível INFO com formato limpo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def _parse_args() -> argparse.Namespace:
    """Define e parseia os argumentos de linha de comando.

    Returns:
        Namespace com os argumentos.
    """
    p = argparse.ArgumentParser(
        description="Pipeline de previsão de média móvel com PyTorch + MT5."
    )
    p.add_argument("--symbol", type=str, default=None, help="Ativo MT5 (ex.: EURUSD)")
    p.add_argument("--timeframe", type=str, default=None, help="Timeframe (ex.: H1)")
    p.add_argument("--ma-period", type=int, default=None, dest="ma_period", help="Período da SMA")
    p.add_argument(
        "--forecast-steps", type=int, default=None, dest="forecast_steps",
        help="Horizonte de previsão (barras)",
    )
    p.add_argument("--bars", type=int, default=None, dest="bars_history", help="Barras históricas")
    p.add_argument(
        "--mode", type=str, default="train", choices=["train", "predict"],
        help="Modo de execução",
    )
    return p.parse_args()


def main() -> int:
    """Função principal: monta config e despacha conforme o modo.

    Returns:
        Código de saída (0 = sucesso, 1 = erro).
    """
    _setup_logging()
    args = _parse_args()
    config = build_config(
        symbol=args.symbol,
        timeframe=args.timeframe,
        ma_period=args.ma_period,
        forecast_steps=args.forecast_steps,
        bars_history=args.bars_history,
    )
    logger = logging.getLogger("main")

    try:
        if args.mode == "train":
            run_pipeline(config)
        else:
            run_dir = build_run_dir(config)
            result = predict_next(run_dir, config)
            _print_forecast(result)
        return 0
    except Exception as exc:  # erro descritivo, sem stacktrace bruto no terminal
        logger.error("❌ Falha no pipeline: %s", exc)
        logging.getLogger(__name__).debug("Detalhe:", exc_info=True)
        return 1


def _print_forecast(result: dict) -> None:
    """Imprime o resultado de previsão formatado no terminal.

    Args:
        result: dicionário retornado por ``predict_next``.
    """
    print("\n" + "=" * 50)
    print(
        f"  PREVISÃO {result['symbol']} {result['timeframe']} "
        f"SMA{result['ma_period']} (+{result['forecast_steps']})"
    )
    print("=" * 50)
    print(f"  MA atual : {result['current_ma']}")
    for f in result["forecast"]:
        print(f"  +{f['bar']} barra(s): {f['ma_value']}")
    print(f"  Direção  : {result['direction']}")
    print(f"  Gerado em: {result['generated_at']}")
    print("=" * 50)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
