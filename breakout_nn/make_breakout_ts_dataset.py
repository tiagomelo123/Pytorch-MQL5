# mt5_export_rates.py
import argparse
import os
from datetime import datetime, timezone
import pandas as pd
import MetaTrader5 as mt5

'''
python make_breakout_ts_dataset.py \
  --input data/EURUSD_H1.csv \
  --output out/eurusd_h1_breakout_ts.csv \
  --depth 100 \
  --expire_bars 12 \
  --shift_points 20 \
  --sl_points 200 \
  --trail_start 120 \
  --trail_dist 120 \
  --trail_step 10 \
  --max_hold_bars 48 \
  --min_r_win 0.20 \
  --point 0.00001
'''

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="H1", choices=["M1","M5","M15","M30","H1","H4","D1"])
    ap.add_argument("--bars", type=int, default=50000, help="quantidade de barras para puxar (do mais recente para trás)")
    ap.add_argument("--output", default="data/EURUSD_H1.csv")
    ap.add_argument("--login", type=int, default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--server", default=None)
    return ap.parse_args()

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

def mt5_init(login=None, password=None, server=None):
    if login and password and server:
        ok = mt5.initialize(login=login, password=password, server=server)
    else:
        ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 initialize() falhou: {mt5.last_error()}")

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    mt5_init(args.login, args.password, args.server)

    symbol = args.symbol
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Falha ao selecionar símbolo {symbol}: {mt5.last_error()}")

    tf = TF_MAP[args.timeframe]

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, args.bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("copy_rates_from_pos retornou vazio. Verifique símbolo/timeframe/barras.")

    df = pd.DataFrame(rates)
    # time vem em epoch seconds
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)  # sem tz no CSV
    # ordem e colunas
    cols = ["time","open","high","low","close","tick_volume","spread","real_volume"]
    df = df[cols]

    df.to_csv(args.output, index=False)
    print(f"OK: {args.output} rows={len(df)} ({symbol} {args.timeframe})")

if __name__ == "__main__":
    main()
