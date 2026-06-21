"""Feature engineering sobre o DataFrame OHLCV bruto.

Todas as features derivadas são normalizadas (em geral por ATR) para garantir
comparabilidade entre ativos e regimes de volatilidade. Nenhuma dependência da
lib ``ta`` — tudo em numpy/pandas puro.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ordem canônica das features de entrada do modelo (X).
# As colunas brutas 'ma' e 'atr' NÃO entram como features (têm unidade de preço).
_BASE_FEATURE_COLS = [
    "log_return",
    "atr_norm",
    "rsi",
    "dist_ma",
    "slope_ma",
    "hl",
    "body",
]
_INTRADAY_FEATURE_COLS = ["sin_hour", "cos_hour"]

_ATR_PERIOD = 14
_RSI_PERIOD = 14


def get_target_columns(forecast_steps: int) -> list[str]:
    """Retorna os nomes das colunas de target multi-step.

    O modelo Seq2Seq prevê ``forecast_steps`` deltas (um por horizonte), então o
    target é uma sequência ``y_step_1 .. y_step_H``.

    Args:
        forecast_steps: horizonte de previsão H.

    Returns:
        Lista ordenada de nomes de colunas de target.
    """
    return [f"y_step_{h}" for h in range(1, forecast_steps + 1)]


def get_feature_columns(timeframe: str) -> list[str]:
    """Retorna a lista ordenada de colunas de feature para o timeframe.

    Em D1 não há variação intradiária, então ``sin_hour``/``cos_hour`` são
    omitidas.

    Args:
        timeframe: código do timeframe (ex.: "H1", "D1").

    Returns:
        Lista de nomes de colunas de feature, na ordem canônica.
    """
    cols = list(_BASE_FEATURE_COLS)
    if timeframe.upper() != "D1":
        cols += _INTRADAY_FEATURE_COLS
    return cols


def _compute_atr(df: pd.DataFrame, period: int = _ATR_PERIOD) -> pd.Series:
    """Calcula o Average True Range (média simples do True Range).

    Args:
        df: DataFrame com colunas ``high``, ``low``, ``close``.
        period: janela do ATR.

    Returns:
        Série do ATR alinhada ao índice do DataFrame.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def _compute_rsi(close: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    """Calcula o RSI de Wilder normalizado para [0, 1].

    Args:
        close: série de preços de fechamento.
        period: janela do RSI.

    Returns:
        Série do RSI dividida por 100 (escala [0, 1]).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Suavização de Wilder via EWM com alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Quando avg_loss == 0 (sem perdas), RSI é 100
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi / 100.0


def build_features(df_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calcula todas as features sobre o DataFrame OHLCV bruto.

    Calcula MA alvo, ATR, e as features normalizadas de entrada, além do target
    ``y`` (delta da MA normalizado por ATR). Remove NaNs gerados pelas janelas
    rolantes e pelo ``shift`` do target.

    Args:
        df_raw: DataFrame com ``time, open, high, low, close, tick_volume``.
        config: dicionário de configuração (ma_period, forecast_steps, timeframe).

    Returns:
        DataFrame com features prontas, colunas auxiliares (``ma``, ``atr``,
        ``ma_target``) e o target ``y``.
    """
    ma_period = config["ma_period"]
    horizon = config["forecast_steps"]
    timeframe = config["timeframe"]

    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # --- Colunas auxiliares (unidade de preço, NÃO entram como features) ---
    df["ma"] = df["close"].rolling(window=ma_period).mean()
    df["atr"] = _compute_atr(df, _ATR_PERIOD)
    # 'ma_target' é um alias explícito da MA usado nos plots
    df["ma_target"] = df["ma"]

    # --- Features de entrada (normalizadas) ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["atr_norm"] = df["atr"] / df["close"]
    df["rsi"] = _compute_rsi(df["close"], _RSI_PERIOD)
    df["dist_ma"] = (df["close"] - df["ma"]) / df["atr"]
    df["slope_ma"] = (df["ma"] - df["ma"].shift(1)) / df["atr"]
    df["hl"] = (df["high"] - df["low"]) / df["atr"]
    df["body"] = (df["close"] - df["open"]) / df["atr"]

    if timeframe.upper() != "D1":
        hour = df["time"].dt.hour
        df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)

    # --- Target multi-step: delta da MA normalizado por ATR, por horizonte ---
    # Cada coluna y_step_h = (ma[t+h] - ma[t]) / atr[t]. Construir todas antes do
    # dropna evita indexar barras futuras após a remoção de NaNs.
    target_cols = get_target_columns(horizon)
    for h, col in enumerate(target_cols, start=1):
        df[col] = (df["ma"].shift(-h) - df["ma"]) / df["atr"]
    # 'y' (escalar) é o delta no horizonte final — usado no histograma do plot 02
    df["y"] = df[target_cols[-1]]

    feature_cols = get_feature_columns(timeframe)
    required = feature_cols + ["ma", "atr", "ma_target", "y"] + target_cols
    df = df.dropna(subset=required).reset_index(drop=True)

    inicio = df["time"].iloc[0].strftime("%Y-%m-%d")
    fim = df["time"].iloc[-1].strftime("%Y-%m-%d")
    logger.info(
        "Features calculadas: %d linhas, %d colunas, target: delta/ATR, período: %s a %s",
        len(df),
        len(feature_cols),
        inicio,
        fim,
    )
    return df
