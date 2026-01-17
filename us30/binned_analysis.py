import numpy as np
import pandas as pd

def binned_analysis(df: pd.DataFrame, prob_col="p_cont", close_col="close",
                    horizon=5, bins=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01)):
    d = df.copy()

    # retorno futuro (log-return)
    d["fwd_ret"] = np.log(d[close_col].shift(-horizon) / d[close_col])

    # binning
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    d["bin"] = pd.cut(d[prob_col], bins=bins, labels=labels, right=False)

    # métricas por bin
    g = d.dropna(subset=["bin", "fwd_ret"]).groupby("bin")["fwd_ret"]

    out = pd.DataFrame({
        "n": g.size(),
        "mean_fwd_ret": g.mean(),
        "median_fwd_ret": g.median(),
        "win_rate": g.apply(lambda x: (x > 0).mean()),
        "p25": g.quantile(0.25),
        "p75": g.quantile(0.75),
    })

    # “edge” anualizado só para leitura (opcional)
    out["mean_fwd_ret_bps"] = out["mean_fwd_ret"] * 10000  # bps
    out = out.sort_index()

    return out