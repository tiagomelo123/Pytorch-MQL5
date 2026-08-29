"""Engenharia de features a partir de uma dataseries OHLCV do MT5."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Nome amigável -> chave interna, usado nos multiselects do painel
FEATURES_DISPONIVEIS = {
    "Retorno (close a close)": "ret_1",
    "Log-retorno": "log_ret_1",
    "Média móvel 10": "sma_10",
    "Média móvel 20": "sma_20",
    "Média móvel 50": "sma_50",
    "Média móvel exp. 12": "ema_12",
    "Média móvel exp. 26": "ema_26",
    "RSI 14": "rsi_14",
    "Volatilidade (desvio padrão 20)": "vol_20",
    "Amplitude (high-low)": "range_hl",
    "Volume normalizado": "vol_norm",
    "Hora do dia (seno/cosseno)": "hora_sin_cos",
    "Dia da semana (seno/cosseno)": "dow_sin_cos",
    "Distância da EMA rápida (20)": "dist_ema_fast",
    "Distância da EMA lenta (50)": "dist_ema_slow",
    "Força da tendência (EMA rápida - lenta)": "trend_strength",
    "Retração vs. último swing (%)": "retracao_pct",
}

# Conjunto sugerido de features para a tarefa de pullback/continuação
FEATURES_PULLBACK_SUGERIDAS = [
    "RSI 14",
    "Distância da EMA rápida (20)",
    "Distância da EMA lenta (50)",
    "Força da tendência (EMA rápida - lenta)",
    "Retração vs. último swing (%)",
    "Volatilidade (desvio padrão 20)",
]

# Conjunto sugerido de features para a tarefa de regime de mercado
FEATURES_REGIME_SUGERIDAS = [
    "RSI 14",
    "Distância da EMA rápida (20)",
    "Distância da EMA lenta (50)",
    "Força da tendência (EMA rápida - lenta)",
    "Volatilidade (desvio padrão 20)",
    "Amplitude (high-low)",
]

# Fórmula de cada coluna final de feature (por nome de coluna, não pelo nome
# amigável) — usado para documentar o pré-processamento no export ONNX, para
# que a mesma conta possa ser replicada em MQL5.
FEATURE_FORMULAS = {
    "ret_1": "close[t]/close[t-1] - 1",
    "log_ret_1": "ln(close[t]/close[t-1])",
    "sma_10": "SMA(close, 10)/close - 1",
    "sma_20": "SMA(close, 20)/close - 1",
    "sma_50": "SMA(close, 50)/close - 1",
    "ema_12": "EMA(close, 12)/close - 1",
    "ema_26": "EMA(close, 26)/close - 1",
    "rsi_14": "RSI de Wilder(close, 14) / 100",
    "vol_20": "desvio padrão móvel(retorno, 20)",
    "range_hl": "(high - low) / close",
    "vol_norm": "(tick_volume - SMA(tick_volume,50)) / desvio_padrão(tick_volume,50)",
    "hora_sin": "sin(2*pi*hora_utc/24)",
    "hora_cos": "cos(2*pi*hora_utc/24)",
    "dow_sin": "sin(2*pi*dia_da_semana/7)  [0=segunda]",
    "dow_cos": "cos(2*pi*dia_da_semana/7)  [0=segunda]",
    "dist_ema_fast": "(close - EMA(close,20)) / close",
    "dist_ema_slow": "(close - EMA(close,50)) / close",
    "trend_strength": "(EMA(close,20) - EMA(close,50)) / close",
    "retracao_pct": (
        "retração atual em relação ao último swing a favor da tendência "
        "(0 fora de candidatos a pullback) — requer replicar a lógica de "
        "swings de core/labeling.py em MQL5; considere não usar esta "
        "feature se o EA não for reproduzir a detecção de swings."
    ),
}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calcula o RSI (Relative Strength Index) clássico de Wilder."""
    delta = close.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.ewm(alpha=1 / period, min_periods=period).mean()
    media_perda = perda.ewm(alpha=1 / period, min_periods=period).mean()
    rs = media_ganho / media_perda.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_features(
    df: pd.DataFrame, feature_keys: list[str], context: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Constrói as colunas de features escolhidas a partir do OHLCV bruto.

    Args:
        df: DataFrame com colunas ``time, open, high, low, close, tick_volume``.
        feature_keys: lista de chaves internas de ``FEATURES_DISPONIVEIS``.
        context: DataFrame opcional (ex.: saída de
            ``labeling.build_pullback_dataset``) com colunas extras
            (``time`` + ex. ``retracao_pct``) para juntar como feature.

    Returns:
        Novo DataFrame contendo ``time``, ``close`` (para referência) e as
        colunas de features solicitadas, sem linhas com NaN.
    """
    out = pd.DataFrame(index=df.index)
    out["time"] = df["time"]
    out["close"] = df["close"]

    close = df["close"]
    ema_fast_20 = close.ewm(span=20, adjust=False).mean()
    ema_slow_50 = close.ewm(span=50, adjust=False).mean()

    if "ret_1" in feature_keys:
        out["ret_1"] = close.pct_change()
    if "log_ret_1" in feature_keys:
        out["log_ret_1"] = np.log(close / close.shift(1))
    if "sma_10" in feature_keys:
        out["sma_10"] = close.rolling(10).mean() / close - 1
    if "sma_20" in feature_keys:
        out["sma_20"] = close.rolling(20).mean() / close - 1
    if "sma_50" in feature_keys:
        out["sma_50"] = close.rolling(50).mean() / close - 1
    if "ema_12" in feature_keys:
        out["ema_12"] = close.ewm(span=12).mean() / close - 1
    if "ema_26" in feature_keys:
        out["ema_26"] = close.ewm(span=26).mean() / close - 1
    if "rsi_14" in feature_keys:
        out["rsi_14"] = _rsi(close, 14) / 100
    if "vol_20" in feature_keys:
        out["vol_20"] = close.pct_change().rolling(20).std()
    if "range_hl" in feature_keys:
        out["range_hl"] = (df["high"] - df["low"]) / close
    if "vol_norm" in feature_keys:
        vol = df["tick_volume"].astype(float)
        out["vol_norm"] = (vol - vol.rolling(50).mean()) / vol.rolling(50).std()
    if "hora_sin_cos" in feature_keys:
        hora = df["time"].dt.hour
        out["hora_sin"] = np.sin(2 * np.pi * hora / 24)
        out["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    if "dow_sin_cos" in feature_keys:
        dow = df["time"].dt.dayofweek
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    if "dist_ema_fast" in feature_keys:
        out["dist_ema_fast"] = (close - ema_fast_20) / close
    if "dist_ema_slow" in feature_keys:
        out["dist_ema_slow"] = (close - ema_slow_50) / close
    if "trend_strength" in feature_keys:
        out["trend_strength"] = (ema_fast_20 - ema_slow_50) / close

    if "retracao_pct" in feature_keys and context is not None and "retracao_pct" in context.columns:
        # Preenche com 0 fora das barras candidatas a pullback (sem retração ativa).
        merge_cols = context[["time", "retracao_pct"]]
        out = out.merge(merge_cols, on="time", how="left")
        out["retracao_pct"] = out["retracao_pct"].fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


def feature_columns(out: pd.DataFrame) -> list[str]:
    """Retorna as colunas de features de um DataFrame construído por ``build_features``."""
    return [c for c in out.columns if c not in ("time", "close")]
