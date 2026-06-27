from pathlib import Path

import pandas as pd

from .config import PRICE_COLUMNS, RAW_DATA_PATH, REQUIRED_COLUMNS


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV nao encontrado em {path}. Coloque o arquivo bruto nesse caminho."
        )

    df = pd.read_csv(path)
    validate_columns(df)

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("A coluna time contem valores que nao puderam ser convertidos.")

    for column in PRICE_COLUMNS + ["volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_numeric = PRICE_COLUMNS + ["volume"]
    if df[invalid_numeric].isna().any().any():
        raise ValueError("Colunas OHLCV contem valores nao numericos ou ausentes.")

    df = (
        df.sort_values("time")
        .drop_duplicates(subset="time", keep="last")
        .reset_index(drop=True)
    )
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")
