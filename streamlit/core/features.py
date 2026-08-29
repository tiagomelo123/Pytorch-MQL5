"""Engenharia de features a partir de uma dataseries OHLCV do MT5."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Nome amigável -> chave interna, usado nos multiselects do painel
FEATURES_DISPONIVEIS = {
    "Retorno (close a close)": "ret_1",
    "Retorno (3 barras)": "ret_3",
    "Retorno (5 barras)": "ret_5",
    "Retorno (10 barras)": "ret_10",
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
    "ATR de Wilder (14, % do preço)": "atr_14_pct",
    "ADX de Wilder (14, força de tendência)": "adx_14",
    "MACD (linha)": "macd_line",
    "MACD (sinal)": "macd_signal",
    "MACD (histograma)": "macd_hist",
    "Inclinação do histograma MACD": "macd_hist_slope",
    "Compressão do MACD (squeeze)": "macd_compress",
    "Bollinger %B": "bb_percent_b",
    "Largura de Bollinger (% do preço)": "bb_width",
    "Z-score do preço vs. SMA 20": "zscore_20",
    "Pavio superior (rejeição, % do range)": "upper_wick_ratio",
    "Pavio inferior (rejeição, % do range)": "lower_wick_ratio",
}

# Conjunto sugerido de features para a tarefa de pullback/continuação
FEATURES_PULLBACK_SUGERIDAS = [
    "RSI 14",
    "Distância da EMA rápida (20)",
    "Distância da EMA lenta (50)",
    "Força da tendência (EMA rápida - lenta)",
    "Retração vs. último swing (%)",
    "Volatilidade (desvio padrão 20)",
    "ADX de Wilder (14, força de tendência)",
]

# Conjunto sugerido de features para a tarefa de regime de mercado
FEATURES_REGIME_SUGERIDAS = [
    "RSI 14",
    "Distância da EMA rápida (20)",
    "Distância da EMA lenta (50)",
    "Força da tendência (EMA rápida - lenta)",
    "Volatilidade (desvio padrão 20)",
    "Amplitude (high-low)",
    "ADX de Wilder (14, força de tendência)",
]

# Conjunto sugerido de features para a tarefa de reversão à média
FEATURES_MEAN_REVERSAL_SUGERIDAS = [
    "Z-score do preço vs. SMA 20",
    "Bollinger %B",
    "RSI 14",
    "ADX de Wilder (14, força de tendência)",
    "Pavio superior (rejeição, % do range)",
    "Pavio inferior (rejeição, % do range)",
    "ATR de Wilder (14, % do preço)",
]

# Fórmula de cada coluna final de feature (por nome de coluna, não pelo nome
# amigável) — usado para documentar o pré-processamento no export ONNX, para
# que a mesma conta possa ser replicada em MQL5.
FEATURE_FORMULAS = {
    "ret_1": "close[t]/close[t-1] - 1",
    "ret_3": "close[t]/close[t-3] - 1",
    "ret_5": "close[t]/close[t-5] - 1",
    "ret_10": "close[t]/close[t-10] - 1",
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
    "atr_14_pct": (
        "ATR de Wilder(14)/close, onde TR = max(high-low, |high-close_ant|, "
        "|low-close_ant|) e Wilder usa média móvel exponencial com alpha=1/14"
    ),
    "adx_14": (
        "ADX de Wilder(14)/100 — mede força da tendência (não direção). "
        "+DM/-DM de high.diff()/-low.diff(); +DI,-DI = 100*RMA(DM,14)/ATR; "
        "DX = 100*|+DI--DI|/(+DI+-DI); ADX = RMA(DX,14)"
    ),
    "macd_line": "(EMA(close,12) - EMA(close,26)) / close",
    "macd_signal": "EMA(linha_MACD_bruta, 9) / close",
    "macd_hist": "(linha_MACD_bruta - sinal_MACD_bruto) / close",
    "macd_hist_slope": "(histograma_bruto[t] - histograma_bruto[t-3]) / close",
    "macd_compress": "|histograma_MACD_bruto| / média_móvel(|histograma_MACD_bruto|, 20)",
    "bb_percent_b": "(close - banda_inferior) / (banda_superior - banda_inferior); bandas = SMA(20) ± 2*desvio_padrão(20)",
    "bb_width": "(banda_superior - banda_inferior) / close",
    "zscore_20": "(close - SMA(close,20)) / desvio_padrão(close,20)",
    "upper_wick_ratio": "(high - max(open, close)) / (high - low)",
    "lower_wick_ratio": "(min(open, close) - low) / (high - low)",
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


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Média móvel de Wilder (RMA): EMA com alpha = 1/period."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range: maior entre (high-low), |high-close_anterior|, |low-close_anterior|."""
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range) com suavização de Wilder."""
    return _wilder_rma(_true_range(df), period)


def _adx_wilder(df: pd.DataFrame, period: int = 14, atr: pd.Series | None = None) -> pd.Series:
    """ADX (Average Directional Index) de Wilder — mede força da tendência (0-100)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    atr = atr if atr is not None else _atr_wilder(df, period)
    plus_di = 100 * _wilder_rma(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_rma(minus_dm, period) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder_rma(dx.fillna(0), period)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD clássico: linha = EMA(fast) - EMA(slow); sinal = EMA(signal) da linha."""
    linha = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sinal = linha.ewm(span=signal, adjust=False).mean()
    return linha, sinal, linha - sinal


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Bandas de Bollinger: SMA(period) ± num_std desvios padrão."""
    media = close.rolling(period).mean()
    desvio = close.rolling(period).std()
    return media + num_std * desvio, media - num_std * desvio


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
    if "ret_3" in feature_keys:
        out["ret_3"] = close.pct_change(3)
    if "ret_5" in feature_keys:
        out["ret_5"] = close.pct_change(5)
    if "ret_10" in feature_keys:
        out["ret_10"] = close.pct_change(10)
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

    precisa_atr = {"atr_14_pct", "adx_14"} & set(feature_keys)
    if precisa_atr:
        atr = _atr_wilder(df, 14)
        if "atr_14_pct" in feature_keys:
            out["atr_14_pct"] = atr / close
        if "adx_14" in feature_keys:
            out["adx_14"] = _adx_wilder(df, 14, atr=atr) / 100

    precisa_macd = {"macd_line", "macd_signal", "macd_hist", "macd_hist_slope", "macd_compress"} & set(feature_keys)
    if precisa_macd:
        linha, sinal, hist = _macd(close)
        if "macd_line" in feature_keys:
            out["macd_line"] = linha / close
        if "macd_signal" in feature_keys:
            out["macd_signal"] = sinal / close
        if "macd_hist" in feature_keys:
            out["macd_hist"] = hist / close
        if "macd_hist_slope" in feature_keys:
            out["macd_hist_slope"] = (hist - hist.shift(3)) / close
        if "macd_compress" in feature_keys:
            out["macd_compress"] = hist.abs() / hist.abs().rolling(20).mean()

    precisa_bb = {"bb_percent_b", "bb_width"} & set(feature_keys)
    if precisa_bb:
        upper, lower = _bollinger(close, 20, 2.0)
        if "bb_percent_b" in feature_keys:
            out["bb_percent_b"] = (close - lower) / (upper - lower)
        if "bb_width" in feature_keys:
            out["bb_width"] = (upper - lower) / close

    if "zscore_20" in feature_keys:
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        out["zscore_20"] = (close - sma_20) / std_20.replace(0, np.nan)

    precisa_wick = {"upper_wick_ratio", "lower_wick_ratio"} & set(feature_keys)
    if precisa_wick:
        corpo_topo = df[["open", "close"]].max(axis=1)
        corpo_base = df[["open", "close"]].min(axis=1)
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        if "upper_wick_ratio" in feature_keys:
            out["upper_wick_ratio"] = (df["high"] - corpo_topo) / candle_range
        if "lower_wick_ratio" in feature_keys:
            out["lower_wick_ratio"] = (corpo_base - df["low"]) / candle_range

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
