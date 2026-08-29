"""Rotulagem de eventos de pullback e continuação/reversão de tendência.

Abordagem baseada em estrutura de mercado (topos e fundos), comum em price
action discricionário, adaptada para gerar rótulos supervisionados:

1. Define a tendência vigente por médias móveis exponenciais (EMA rápida x
   EMA lenta).
2. Detecta topos e fundos (swings) por comparação local (máximo/mínimo em
   uma janela de N barras para cada lado).
3. Marca como **candidato a pullback** qualquer barra em que o preço esteja
   retraindo contra a tendência vigente, a partir do último swing a favor da
   tendência, além de uma retração mínima.
4. Olha ``horizon`` barras à frente: se o preço romper o extremo do swing a
   favor da tendência sem antes romper a estrutura (o swing contrário), o
   pullback é rotulado como **continuação (1)**; caso contrário, como
   **reversão/pullback sem continuação (0)**.

Importante: os swings só são considerados "confirmados" ``swing_order``
barras depois de ocorrerem (é preciso ver as barras seguintes para saber que
aquele ponto foi de fato um extremo local) — isso evita "olhar o futuro" na
hora de definir o contexto de cada barra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_trend_ema(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50) -> pd.DataFrame:
    """Calcula EMAs rápida/lenta e a tendência vigente por cruzamento delas.

    Returns:
        DataFrame com colunas ``ema_fast``, ``ema_slow`` e ``trend``
        (``1`` = alta, ``-1`` = baixa, ``0`` = indefinida/lateral).
    """
    close = df["close"]
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    trend = pd.Series(np.where(ema_f > ema_s, 1, np.where(ema_f < ema_s, -1, 0)), index=df.index)
    return pd.DataFrame({"ema_fast": ema_f, "ema_slow": ema_s, "trend": trend})


def detect_swings(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """Detecta topos e fundos locais (swing highs/lows) por comparação em janela.

    Uma barra é um swing high se seu ``high`` é o máximo entre ``order``
    barras antes e ``order`` barras depois (idem para swing low com ``low``).

    Args:
        df: DataFrame com colunas ``high``, ``low``.
        order: número de barras de cada lado usadas para confirmar o extremo.

    Returns:
        DataFrame com colunas booleanas ``swing_high`` e ``swing_low``,
        indexado como ``df``.
    """
    janela = 2 * order + 1
    high_max = df["high"].rolling(janela, center=True).max()
    low_min = df["low"].rolling(janela, center=True).min()
    swing_high = (df["high"] == high_max).fillna(False)
    swing_low = (df["low"] == low_min).fillna(False)
    return pd.DataFrame({"swing_high": swing_high, "swing_low": swing_low})


def build_pullback_dataset(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    swing_order: int = 5,
    horizon: int = 20,
    min_retracement: float = 0.001,
) -> pd.DataFrame:
    """Constrói o dataset de candidatos a pullback e seus rótulos.

    Args:
        df: OHLCV bruto (colunas ``time, open, high, low, close, ...``).
        ema_fast: período da EMA rápida (define a tendência).
        ema_slow: período da EMA lenta (define a tendência).
        swing_order: barras de cada lado para confirmar um swing high/low.
        horizon: barras à frente observadas para decidir se o pullback
            "continuou" a tendência ou não.
        min_retracement: retração mínima (fração do preço, ex.: 0.001 = 0.1%)
            em relação ao último swing a favor da tendência para considerar
            a barra como candidata a pullback.

    Returns:
        DataFrame indexado como ``df`` com colunas:
        ``time``, ``trend`` (1/-1/0), ``is_candidate`` (bool),
        ``label`` (1.0 = continuação, 0.0 = reversão/sem continuação,
        ``NaN`` para barras que não são candidatas), ``retracao_pct``
        (retração no momento da barra, só para candidatas).
    """
    trend_df = detect_trend_ema(df, ema_fast, ema_slow)
    swings = detect_swings(df, swing_order)

    trend = trend_df["trend"].to_numpy()
    swing_high = swings["swing_high"].to_numpy()
    swing_low = swings["swing_low"].to_numpy()
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    # Swings só são "conhecidos" swing_order barras depois de ocorrerem.
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    cur_high, cur_low = np.nan, np.nan
    for i in range(n):
        j = i - swing_order
        if j >= 0:
            if swing_high[j]:
                cur_high = high[j]
            if swing_low[j]:
                cur_low = low[j]
        last_swing_high[i] = cur_high
        last_swing_low[i] = cur_low

    is_candidate = np.zeros(n, dtype=bool)
    label = np.full(n, np.nan)
    retracao_pct = np.full(n, np.nan)

    limite = n - horizon
    for i in range(swing_order, max(swing_order, limite)):
        t = trend[i]
        swing_hi, swing_lo = last_swing_high[i], last_swing_low[i]
        if np.isnan(swing_hi) or np.isnan(swing_lo):
            continue

        if t == 1 and close[i] < swing_hi:
            retracao = (swing_hi - close[i]) / swing_hi
            if retracao > min_retracement:
                is_candidate[i] = True
                retracao_pct[i] = retracao
                fut_high = high[i + 1 : i + 1 + horizon].max()
                fut_low = low[i + 1 : i + 1 + horizon].min()
                rompeu_estrutura = fut_low < swing_lo
                continuou = (fut_high > swing_hi) and not rompeu_estrutura
                label[i] = 1.0 if continuou else 0.0

        elif t == -1 and close[i] > swing_lo:
            retracao = (close[i] - swing_lo) / swing_lo
            if retracao > min_retracement:
                is_candidate[i] = True
                retracao_pct[i] = retracao
                fut_high = high[i + 1 : i + 1 + horizon].max()
                fut_low = low[i + 1 : i + 1 + horizon].min()
                rompeu_estrutura = fut_high > swing_hi
                continuou = (fut_low < swing_lo) and not rompeu_estrutura
                label[i] = 1.0 if continuou else 0.0

    return pd.DataFrame(
        {
            "time": df["time"].reset_index(drop=True),
            "trend": trend,
            "is_candidate": is_candidate,
            "label": label,
            "retracao_pct": retracao_pct,
        }
    )


def build_market_regime_labels(
    df: pd.DataFrame,
    horizon: int = 20,
    vol_window: int = 20,
    k_lateral: float = 1.0,
) -> pd.DataFrame:
    """Rotula o regime de mercado (baixa/lateral/alta) nas próximas ``horizon`` barras.

    O retorno esperado por puro acaso (ruído) cresce com a raiz do horizonte
    (``vol * sqrt(horizon)``, escala de um passeio aleatório). Por isso o
    limiar que separa "lateral" de uma tendência real é adaptativo: usa a
    volatilidade recente (desvio padrão dos retornos) em vez de uma
    porcentagem fixa — assim funciona de forma parecida em ativos/timeframes
    com volatilidades bem diferentes.

    Classes (índice usado no treino): ``0 = Baixa``, ``1 = Lateral``,
    ``2 = Alta`` (ver ``config.REGIME_CLASSES``).

    Args:
        df: OHLCV bruto.
        horizon: barras à frente cujo retorno define o regime.
        vol_window: janela (em barras) usada para estimar a volatilidade
            recente (desvio padrão dos retornos), base do limiar adaptativo.
        k_lateral: multiplicador do limiar — quanto maior, mais amplo o
            intervalo de retorno considerado "lateral" (sem tendência clara).

    Returns:
        DataFrame com colunas ``time``, ``label`` (0/1/2, ``NaN`` nas últimas
        ``horizon`` barras, onde o retorno futuro não é conhecido) e
        ``limiar`` (limiar de retorno usado naquela barra, para referência).
    """
    close = df["close"]
    vol = close.pct_change().rolling(vol_window).std()
    limiar = k_lateral * vol * np.sqrt(horizon)
    fut_ret = close.shift(-horizon) / close - 1

    label = pd.Series(np.nan, index=df.index)
    valido = fut_ret.notna() & limiar.notna()
    label[valido & (fut_ret > limiar)] = 2.0  # Alta
    label[valido & (fut_ret < -limiar)] = 0.0  # Baixa
    label[valido & (fut_ret >= -limiar) & (fut_ret <= limiar)] = 1.0  # Lateral

    return pd.DataFrame(
        {
            "time": df["time"].reset_index(drop=True),
            "label": label.reset_index(drop=True),
            "limiar": limiar.reset_index(drop=True),
        }
    )
