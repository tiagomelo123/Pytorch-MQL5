"""Entry point CLI do pipeline de previsão de média móvel.

Sem lógica de negócio aqui: apenas parse de argumentos e despacho para o
``pipeline.runner``.
"""

import argparse
import logging
import sys

from config import build_config
from pipeline.runner import run_pipeline


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
        "--retrain", action="store_true", default=False,
        help="Força novo treino mesmo se modelo existir (sobrescreve artefatos)",
    )
    return p.parse_args()


def main() -> int:
    """Função principal: monta config e despacha para o runner.

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
        run_pipeline(config, retrain=args.retrain)
        return 0
    except Exception as exc:  # erro descritivo, sem stacktrace bruto no terminal
        logger.error("❌ Falha no pipeline: %s", exc)
        logging.getLogger(__name__).debug("Detalhe:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
