import argparse
import pandas as pd
from pathlib import Path
from binned_analysis import binned_analysis  # seu script atual

import argparse
import os
import pandas as pd

from binned_analysis import binned_analysis

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / os.path.join("out", "us30_h1_nn_probs.csv")
"""
    Executa Binned Analysis a partir de um CSV contendo:
    - time   : timestamp do candle
    - close  : preço de fechamento
    - p_cont : probabilidade de continuação (NN)

    Se --csv não for informado, utiliza:
        out/us30_h1_nn_probs.csv
    """

def main():
    

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Caminho do CSV com probabilidades (default: {DEFAULT_CSV})",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Quantidade de candles à frente para medir retorno",
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Arquivo de saída (default: out/binned_h{horizon}.csv)",
    )

    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Arquivo não encontrado: {args.csv}")

    # saída padrão automática
    if args.out_csv is None:
        args.out_csv = os.path.join("out", f"binned_h{args.horizon}.csv")

    df = pd.read_csv(args.csv)
    df["time"] = pd.to_datetime(df["time"])

    report = binned_analysis(
        df,
        prob_col="p_cont",
        close_col="close",
        horizon=args.horizon,
        bins=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01),
    )

    print("\n=== BINNED ANALYSIS ===")
    print(report)
    report.to_csv(args.out_csv, index=True)
    print(f"\nOK: {args.out_csv}")


if __name__ == "__main__":
    main()
