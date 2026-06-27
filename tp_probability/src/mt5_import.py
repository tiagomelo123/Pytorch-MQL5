import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import SYMBOL, TIMEFRAME, normalize_symbol, normalize_timeframe, raw_data_path


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M2": "TIMEFRAME_M2",
    "M3": "TIMEFRAME_M3",
    "M4": "TIMEFRAME_M4",
    "M5": "TIMEFRAME_M5",
    "M6": "TIMEFRAME_M6",
    "M10": "TIMEFRAME_M10",
    "M12": "TIMEFRAME_M12",
    "M15": "TIMEFRAME_M15",
    "M20": "TIMEFRAME_M20",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H2": "TIMEFRAME_H2",
    "H3": "TIMEFRAME_H3",
    "H4": "TIMEFRAME_H4",
    "H6": "TIMEFRAME_H6",
    "H8": "TIMEFRAME_H8",
    "H12": "TIMEFRAME_H12",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}


def import_from_mt5(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    bars: int = 5000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    mt5 = import_metatrader5()
    mt5_timeframe = resolve_mt5_timeframe(mt5, timeframe)
    normalized_symbol = normalize_symbol(symbol)
    output = output_path or raw_data_path(normalized_symbol, timeframe)

    if not mt5.initialize():
        code, message = mt5.last_error()
        raise RuntimeError(f"Falha ao inicializar MetaTrader5: {code} - {message}")

    try:
        if not mt5.symbol_select(normalized_symbol, True):
            code, message = mt5.last_error()
            raise RuntimeError(
                f"Nao foi possivel selecionar {normalized_symbol}: {code} - {message}"
            )

        rates = mt5.copy_rates_from_pos(normalized_symbol, mt5_timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            code, message = mt5.last_error()
            raise RuntimeError(
                f"Nenhum candle retornado para {normalized_symbol} {timeframe}: "
                f"{code} - {message}"
            )
    finally:
        mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("time").drop_duplicates(subset="time", keep="last")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def import_metatrader5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise ImportError(
            "Pacote MetaTrader5 nao instalado. Rode: python -m pip install MetaTrader5"
        ) from exc
    return mt5


def resolve_mt5_timeframe(mt5, timeframe: str) -> int:
    normalized = normalize_timeframe(timeframe)
    attr_name = TIMEFRAME_MAP.get(normalized)
    if attr_name is None:
        supported = ", ".join(TIMEFRAME_MAP)
        raise ValueError(f"Timeframe {timeframe} nao suportado. Use um de: {supported}")
    return getattr(mt5, attr_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa candles direto do MetaTrader 5.")
    parser.add_argument("--symbol", default=SYMBOL, help="Ativo. Ex: EURUSD, GBPUSD, USDJPY.")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe. Ex: M1, M5, M15, H1.")
    parser.add_argument("--bars", type=int, default=5000, help="Quantidade de candles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    output = raw_data_path(args.symbol, args.timeframe)
    df = import_from_mt5(args.symbol, args.timeframe, args.bars, output)
    print(f"Importacao iniciada em UTC: {started_at:%Y-%m-%d %H:%M:%S}")
    print(f"Candles importados: {len(df)}")
    print(f"Arquivo salvo em: {output}")
    print(f"Periodo: {df['time'].min()} -> {df['time'].max()}")


if __name__ == "__main__":
    main()
