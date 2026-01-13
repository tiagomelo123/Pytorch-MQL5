import os
import sys
import time
import numpy as np
import pandas as pd

import MetaTrader5 as mt5


# =========================================================
# 1) Indicadores (pandas/numpy puro)
# =========================================================

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rma(s: pd.Series, period: int) -> pd.Series:
    # Wilder smoothing (RMA)
    return s.ewm(alpha=1/period, adjust=False).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return rma(tr, period)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    macd_signal = ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_rma = rma(tr, period)

    plus_di = 100 * rma(pd.Series(plus_dm, index=high.index), period) / atr_rma
    minus_di = 100 * rma(pd.Series(minus_dm, index=high.index), period) / atr_rma

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx, period)

def bb_width(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return (upper - lower) / close.replace(0, np.nan)


# =========================================================
# 2) Features
# =========================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Log returns
    log_close = np.log(out["close"].replace(0, np.nan))
    for k in range(1, 8):
        out[f"ret_{k}"] = log_close.diff(k)

    out["ret_sum_3"] = out["ret_1"] + out["ret_2"] + out["ret_3"]
    out["ret_vol_7"] = out[[f"ret_{k}" for k in range(1, 8)]].std(axis=1)

    # EMAs
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["ema50_slope_5"] = out["ema50"] - out["ema50"].shift(5)

    # MACD
    macd_line, macd_signal, macd_hist = macd(out["close"], 12, 26, 9)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["macd_hist_slope_1"] = out["macd_hist"] - out["macd_hist"].shift(1)
    out["macd_hist_slope_3"] = out["macd_hist"] - out["macd_hist"].shift(3)

    # ATR / regime
    out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    out["atr_pct"] = out["atr14"] / out["close"].replace(0, np.nan)
    out["dist_close_ema50"] = (out["close"] - out["ema50"]) / out["atr14"].replace(0, np.nan)

    # ADX
    out["adx14"] = adx(out["high"], out["low"], out["close"], 14)

    # Bollinger width
    out["bb_width_20"] = bb_width(out["close"], 20, 2.0)

    # MACD compress
    out["macd_hist_abs"] = out["macd_hist"].abs()
    out["macd_hist_abs_ma20"] = out["macd_hist_abs"].rolling(20).mean()
    out["macd_compress"] = out["macd_hist_abs"] / out["macd_hist_abs_ma20"].replace(0, np.nan)

    return out


# =========================================================
# 3) Labels: CONT vs REV (apenas tendência de ALTA)
# =========================================================

def add_labels_cont_rev(
    df_feat: pd.DataFrame,
    X: int = 8,
    thr_atr: float = 0.4,
    use_adx: bool = True,
    adx_min: float = 20.0,
) -> pd.DataFrame:
    out = df_feat.copy()

    trend_up = (out["ema50"] > out["ema200"]) & (out["ema50_slope_5"] > 0) & (out["macd_hist"] > 0)
    if use_adx:
        trend_up = trend_up & (out["adx14"] >= adx_min)

    delta_fwd = out["close"].shift(-X) - out["close"]
    thr = thr_atr * out["atr14"]

    y = pd.Series(np.nan, index=out.index)
    y.loc[trend_up & (delta_fwd >= thr)] = 1  # CONT
    y.loc[trend_up & (delta_fwd <= -thr)] = 0 # REV

    out["label"] = y
    return out


# =========================================================
# 4) MT5: conexão + download candles
# =========================================================

def mt5_connect(login=None, password=None, server=None, path=None) -> None:
    """
    Conecta no terminal MT5.
    - Se MT5 já estiver aberto e logado, mt5.initialize() geralmente basta.
    - Se quiser, passe login/password/server.
    """
    kwargs = {}
    if path:
        kwargs["path"] = path
    if login is not None:
        kwargs["login"] = int(login)
    if password is not None:
        kwargs["password"] = str(password)
    if server is not None:
        kwargs["server"] = str(server)

    ok = mt5.initialize(**kwargs)
    if not ok:
        raise RuntimeError(f"MT5 initialize() falhou: {mt5.last_error()}")

def find_us30_symbol(prefer: str | None = None) -> str:
    """
    Tenta achar o símbolo do US30 (varia por corretora).
    Se 'prefer' for passado e existir, usa ele.
    """
    symbols = mt5.symbols_get()
    if symbols is None:
        raise RuntimeError(f"symbols_get() falhou: {mt5.last_error()}")

    names = [s.name for s in symbols]

    if prefer and prefer in names:
        return prefer

    # Heurísticas comuns
    candidates = []
    keys = ["US30", "DJI", "DOW", "WALL", "WS30", "US_30", "USA30"]
    for n in names:
        up = n.upper()
        if any(k in up for k in keys):
            candidates.append(n)

    if not candidates:
        # fallback: tenta "US30" direto
        if "US30" in names:
            return "US30"
        raise RuntimeError("Não encontrei um símbolo parecido com US30/DJI/DOW. "
                           "Passe o nome exato do símbolo via prefer=...")

    # Prioriza os que começam com US30
    candidates.sort(key=lambda x: (0 if x.upper().startswith("US30") else 1, len(x)))
    return candidates[0]

def ensure_symbol_selected(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info({symbol}) retornou None. last_error={mt5.last_error()}")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Falha ao selecionar símbolo {symbol}. last_error={mt5.last_error()}")

def fetch_rates(symbol: str, timeframe, n_bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos falhou/zerado. last_error={mt5.last_error()}")

    df = pd.DataFrame(rates)
    # MT5 devolve time em epoch (segundos)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").set_index("time")
    return df


# =========================================================
# 5) Pipeline
# =========================================================

def main():
    # ---------- CONFIG ----------
    N_BARS = 10000            # ajuste (ex.: 5000 ~ 208 dias em H1)
    TIMEFRAME = mt5.TIMEFRAME_H1

    # Se quiser forçar o símbolo exato da sua corretora, preencha aqui:
    FORCE_SYMBOL = None      # ex.: "US30.cash" ou "WallStreet30"

    # Labels (opcional)
    MAKE_LABELS = True
    X = 8
    THR_ATR = 0.4
    USE_ADX = True
    ADX_MIN = 20.0

    # Saídas
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    ohlc_csv = os.path.join(out_dir, "us30_h1_ohlc.csv")
    feat_csv = os.path.join(out_dir, "us30_h1_features.csv")
    dataset_csv = os.path.join(out_dir, "us30_h1_macd_dataset.csv")

    # ---------- CONNECT ----------
    mt5_connect()  # assume MT5 aberto/logado

    # ---------- SYMBOL ----------
    symbol = find_us30_symbol(prefer=FORCE_SYMBOL)
    ensure_symbol_selected(symbol)
    print(f"Usando símbolo: {symbol}")

    # ---------- FETCH ----------
    df = fetch_rates(symbol, TIMEFRAME, N_BARS)

    # Mantém só OHLC (e opcionalmente spread/tick_volume se quiser)
    df_ohlc = df[["open", "high", "low", "close"]].copy()
    df_ohlc.to_csv(ohlc_csv, index=True)
    print(f"Saved OHLC: {ohlc_csv} | bars={len(df_ohlc)}")

    # ---------- FEATURES ----------
    feat = build_features(df_ohlc)
    feat.to_csv(feat_csv, index=True)
    print(f"Saved features (full): {feat_csv}")

    # ---------- DATASET FINAL ----------
    FEATURE_COLS = (
        [f"ret_{k}" for k in range(1, 8)]
        + ["ret_sum_3", "ret_vol_7"]
        + ["macd_hist", "macd_hist_slope_1", "macd_hist_slope_3"]
        + ["ema50_slope_5", "dist_close_ema50"]
        + ["adx14", "atr_pct", "bb_width_20", "macd_compress"]
    )

    if MAKE_LABELS:
        feat = add_labels_cont_rev(
            feat,
            X=X,
            thr_atr=THR_ATR,
            use_adx=USE_ADX,
            adx_min=ADX_MIN,
        )
        dataset = feat[FEATURE_COLS + ["label"]].dropna()
        dataset.to_csv(dataset_csv, index=True)

        vc = dataset["label"].value_counts()
        print(f"Saved dataset: {dataset_csv} | samples={len(dataset)}")
        print("Label counts:\n", vc.to_string())
    else:
        dataset = feat[FEATURE_COLS].dropna()
        dataset.to_csv(dataset_csv, index=True)
        print(f"Saved dataset (no labels): {dataset_csv} | samples={len(dataset)}")

    mt5.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERRO:", str(e))
        try:
            mt5.shutdown()
        except Exception:
            pass
        sys.exit(1)