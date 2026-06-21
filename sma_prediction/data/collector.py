"""Coleta de dados OHLCV diretamente do terminal MetaTrader 5."""

import logging
import os

import pandas as pd

try:  # MetaTrader5 só está disponível no Windows com o terminal instalado
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - ambiente sem MT5
    mt5 = None

logger = logging.getLogger(__name__)

# Mapeamento de timeframes textuais para as constantes do MT5
_TF_NAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def _resolve_timeframe(timeframe: str) -> int:
    """Converte o timeframe textual para a constante numérica do MT5.

    Args:
        timeframe: código do timeframe (ex.: "H1").

    Returns:
        Constante inteira correspondente do MetaTrader5.

    Raises:
        ValueError: se o timeframe não for suportado.
    """
    key = timeframe.upper()
    if key not in _TF_NAMES:
        raise ValueError(
            f"Timeframe inválido: '{timeframe}'. "
            f"Use um de: {', '.join(_TF_NAMES)}"
        )
    return getattr(mt5, _TF_NAMES[key])


def _validate_symbol(symbol: str) -> None:
    """Verifica se o símbolo existe no MT5 e o seleciona no Market Watch.

    Args:
        symbol: nome do ativo (ex.: "EURUSD").

    Raises:
        ValueError: se o símbolo não existir, sugerindo símbolos similares.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        prefix = symbol[:3].upper()
        all_symbols = mt5.symbols_get()
        similares = [
            s.name for s in (all_symbols or []) if s.name.upper().startswith(prefix)
        ][:10]
        sugestao = f" Símbolos similares: {similares}" if similares else ""
        raise ValueError(f"Símbolo '{symbol}' não encontrado no MT5.{sugestao}")

    if not info.visible:
        # Tenta tornar o símbolo visível no Market Watch para permitir coleta
        if not mt5.symbol_select(symbol, True):
            raise ValueError(
                f"Não foi possível selecionar o símbolo '{symbol}' no Market Watch."
            )


def collect_ohlcv(config: dict, run_dir: str) -> pd.DataFrame:
    """Coleta barras OHLCV do MT5 e salva o dataset bruto.

    Verifica a conexão com o terminal, valida o símbolo, baixa o histórico,
    remove barras com volume zero e salva o resultado em
    ``run_dir/dataset_raw.csv``.

    Args:
        config: dicionário de configuração (symbol, timeframe, bars_history,
            lookback_window, ma_period, forecast_steps).
        run_dir: diretório de artefatos do run atual.

    Returns:
        DataFrame com colunas ``time, open, high, low, close, tick_volume``.

    Raises:
        RuntimeError: se o MT5 não estiver disponível ou a inicialização falhar.
        ValueError: se o símbolo for inválido ou houver barras insuficientes.
    """
    if mt5 is None:
        raise RuntimeError(
            "Pacote MetaTrader5 não instalado. Rode 'pip install MetaTrader5'."
        )

    symbol = config["symbol"]
    timeframe = config["timeframe"]
    bars = config["bars_history"]

    if not mt5.initialize():
        code, msg = mt5.last_error()
        raise RuntimeError(
            f"Falha ao inicializar o MT5 ({code}: {msg}). "
            "Verifique se o terminal está aberto e com conta logada."
        )

    try:
        _validate_symbol(symbol)
        tf = _resolve_timeframe(timeframe)

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise ValueError(
                f"Nenhuma barra retornada para {symbol} {timeframe} ({code}: {msg})."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()

        # Remove barras de fim de semana / feriados (volume zero)
        antes = len(df)
        df = df[df["tick_volume"] > 0].reset_index(drop=True)
        removidas = antes - len(df)

        # A última barra pode estar em andamento; descartar para usar só fechadas
        df = df.iloc[:-1].reset_index(drop=True)

        minimo = (
            config["lookback_window"]
            + config["ma_period"]
            + config["forecast_steps"]
            + 100
        )
        if len(df) < minimo:
            raise ValueError(
                f"Barras insuficientes após limpeza: {len(df)} < {minimo} necessárias. "
                "Aumente BARS_HISTORY ou use um timeframe menor."
            )

        os.makedirs(run_dir, exist_ok=True)
        raw_path = os.path.join(run_dir, "dataset_raw.csv")
        df.to_csv(raw_path, index=False)

        inicio = df["time"].iloc[0].strftime("%Y-%m-%d")
        fim = df["time"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(
            "%d barras válidas coletadas (%s a %s) | %d barras removidas (volume zero)",
            len(df),
            inicio,
            fim,
            removidas,
        )
        logger.info("Salvo: %s", raw_path)
        return df
    finally:
        mt5.shutdown()
