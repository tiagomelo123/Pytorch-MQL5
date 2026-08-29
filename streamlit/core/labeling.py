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

from core.features import _adx_wilder, _atr_wilder


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


def build_mean_reversal_dataset(
    df: pd.DataFrame,
    zscore_window: int = 20,
    zscore_threshold: float = 2.0,
    use_adx_filter: bool = True,
    adx_max: float = 20.0,
    atr_period: int = 14,
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    horizon: int = 20,
) -> pd.DataFrame:
    """Constrói o dataset de candidatos a reversão à média e seus rótulos.

    Detecção do "setup" (barra esticada): o preço está a mais de
    ``zscore_threshold`` desvios padrão da SMA(``zscore_window``) — z-score
    positivo = esticado para cima (candidato a venda/reversão para baixo),
    z-score negativo = esticado para baixo (candidato a compra/reversão para
    cima). Opcionalmente, só considera candidatos quando o ADX de Wilder
    está abaixo de ``adx_max`` (mercado sem tendência forte, mais propício a
    reversões).

    Rótulo (barreira tripla baseada em ATR): a partir da barra candidata,
    define um alvo (TP) a ``tp_atr_mult`` × ATR em direção à média e um stop
    (SL) a ``sl_atr_mult`` × ATR na direção contrária (continuação da
    extensão). Observa até ``horizon`` barras à frente, barra a barra:
    ``label = 1`` se o TP for tocado antes do SL, ``label = 0`` se o SL for
    tocado antes (ou se ambos forem tocados na mesma barra — resultado
    ambíguo tratado de forma conservadora — ou se nenhum dos dois for
    tocado dentro do horizonte).

    Args:
        df: OHLCV bruto (colunas ``time, open, high, low, close, ...``).
        zscore_window: janela (barras) da SMA/desvio padrão do z-score.
        zscore_threshold: |z-score| mínimo para considerar a barra esticada.
        use_adx_filter: se ``True``, só marca candidatos com ADX < ``adx_max``.
        adx_max: limite de ADX (força de tendência) para o filtro acima.
        atr_period: período do ATR de Wilder usado para o TP/SL.
        tp_atr_mult: múltiplo do ATR até o alvo (na direção da média).
        sl_atr_mult: múltiplo do ATR até o stop (na direção contrária).
        horizon: barras à frente observadas para checar TP/SL.

    Returns:
        DataFrame indexado como ``df`` com colunas: ``time``, ``zscore``,
        ``is_candidate`` (bool), ``direction`` (``1`` = esperando alta/
        reversão de queda, ``-1`` = esperando baixa/reversão de alta, ``0``
        fora de candidatos), ``label`` (1.0 = TP antes do SL, 0.0 = SL antes
        ou sem resolução, ``NaN`` para barras que não são candidatas).
    """
    close = df["close"]
    sma = close.rolling(zscore_window).mean()
    desvio = close.rolling(zscore_window).std()
    zscore = (close - sma) / desvio.replace(0, np.nan)

    atr = _atr_wilder(df, atr_period)
    adx = _adx_wilder(df, atr_period, atr=atr) if use_adx_filter else None

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)
    atr_arr = atr.to_numpy(dtype=float)
    z_arr = zscore.to_numpy(dtype=float)
    adx_arr = adx.to_numpy(dtype=float) if adx is not None else None
    n = len(df)

    is_candidate = np.zeros(n, dtype=bool)
    direction = np.zeros(n)
    label = np.full(n, np.nan)

    limite = n - horizon
    inicio = max(zscore_window, atr_period)
    for i in range(inicio, max(inicio, limite)):
        z = z_arr[i]
        a = atr_arr[i]
        if np.isnan(z) or np.isnan(a) or a <= 0 or abs(z) < zscore_threshold:
            continue
        if use_adx_filter:
            adx_i = adx_arr[i]
            if np.isnan(adx_i) or adx_i > adx_max:
                continue

        entry = close_arr[i]
        if z > 0:
            dir_i = -1.0  # esticado para cima -> espera reversão para baixo
            tp_price = entry - tp_atr_mult * a
            sl_price = entry + sl_atr_mult * a
        else:
            dir_i = 1.0  # esticado para baixo -> espera reversão para cima
            tp_price = entry + tp_atr_mult * a
            sl_price = entry - sl_atr_mult * a

        is_candidate[i] = True
        direction[i] = dir_i
        resultado = 0.0
        for j in range(i + 1, i + 1 + horizon):
            if dir_i < 0:
                tp_hit = low[j] <= tp_price
                sl_hit = high[j] >= sl_price
            else:
                tp_hit = high[j] >= tp_price
                sl_hit = low[j] <= sl_price
            if tp_hit and sl_hit:
                resultado = 0.0  # ambíguo na mesma barra -> conservador
                break
            if tp_hit:
                resultado = 1.0
                break
            if sl_hit:
                resultado = 0.0
                break
        label[i] = resultado

    return pd.DataFrame(
        {
            "time": df["time"].reset_index(drop=True),
            "zscore": zscore.reset_index(drop=True),
            "is_candidate": is_candidate,
            "direction": direction,
            "label": label,
        }
    )
