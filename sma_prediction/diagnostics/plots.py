"""Geração de todos os gráficos de diagnóstico do pipeline."""

import logging
import os

import matplotlib

matplotlib.use("Agg")  # backend não interativo (salva PNGs sem display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.features import get_feature_columns
from diagnostics.learning_check import diagnose_learning

logger = logging.getLogger(__name__)


def _run_id(config: dict) -> str:
    """Monta o identificador legível do run para títulos.

    Args:
        config: dicionário de configuração.

    Returns:
        String tipo ``EURUSD | H1 | SMA20 | Forecast +5``.
    """
    return (
        f"{config['symbol']} | {config['timeframe']} | "
        f"SMA{config['ma_period']} | Forecast +{config['forecast_steps']}"
    )


def _apply_style(config: dict) -> None:
    """Aplica o estilo matplotlib configurado (com fallback)."""
    try:
        plt.style.use(config.get("plot_style", "seaborn-v0_8-darkgrid"))
    except OSError:  # estilo indisponível na versão do matplotlib
        plt.style.use("ggplot")


def _save(fig, run_dir: str, name: str, dpi: int) -> str:
    """Salva a figura em ``run_dir/plots/name`` e fecha-a.

    Args:
        fig: figura matplotlib.
        run_dir: diretório de artefatos.
        name: nome do arquivo PNG.
        dpi: resolução.

    Returns:
        Caminho do arquivo salvo.
    """
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("✅ %s", name)
    return path


# --------------------------------------------------------------------------- #
# Plot 01 — Preço e MA histórica
# --------------------------------------------------------------------------- #
def plot_raw_price(df: pd.DataFrame, splits: dict, config: dict, run_dir: str) -> str:
    """Plota close + MA histórica e volume, marcando os cortes de split."""
    _apply_style(config)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    t = pd.to_datetime(df["time"])
    ax1.plot(t, df["close"], color="lightgray", lw=0.8, label="Close")
    ax1.plot(t, df["ma_target"], color="tab:blue", lw=1.2, label="MA alvo")

    train_end, val_end = splits["train_end"], splits["val_end"]
    for idx, lbl, c in [(train_end, "início val", "green"), (val_end, "início test", "red")]:
        if 0 <= idx < len(t):
            ax1.axvline(t.iloc[idx], color=c, ls=":", lw=1.5, label=f"{lbl}: {t.iloc[idx].date()}")

    ax1.set_title(f"Preço e MA histórica — {_run_id(config)}")
    ax1.set_ylabel("Preço")
    ax1.legend(loc="best", fontsize=8)

    ax2.bar(t, df["tick_volume"], color="tab:purple", width=0.02)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Tempo")
    return _save(fig, run_dir, "01_raw_price.png", config["plot_dpi"])


# --------------------------------------------------------------------------- #
# Plot 02 — Painel de features
# --------------------------------------------------------------------------- #
def plot_features(df: pd.DataFrame, config: dict, run_dir: str) -> str:
    """Plota cada feature ativa + distribuição do target ``y``."""
    _apply_style(config)
    feature_cols = get_feature_columns(config["timeframe"])
    panels = feature_cols + ["y"]
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    t = pd.to_datetime(df["time"])

    for i, col in enumerate(panels):
        ax = axes[i]
        serie = df[col]
        nan_pct = serie.isna().mean() * 100
        if col == "y":
            ax.hist(serie.dropna(), bins=60, color="tab:orange", alpha=0.8)
            ax.axvline(0.0, color="black", ls="--", lw=1)
            ax.set_title(f"target y (delta/ATR)\nμ={serie.mean():.3f} σ={serie.std():.3f}")
        else:
            ax.plot(t, serie, lw=0.7, color="tab:blue")
            ref = 0.5 if col == "rsi" else 0.0
            ax.axhline(ref, color="gray", ls="--", lw=0.8)
            ax.set_title(f"{col}\nμ={serie.mean():.3f} σ={serie.std():.3f}")
        if nan_pct > 1.0:
            ax.text(
                0.5, 0.5, f"AVISO: {nan_pct:.1f}% NaN", color="red", fontsize=11,
                ha="center", va="center", transform=ax.transAxes, fontweight="bold",
            )

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Painel de features — {_run_id(config)}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, run_dir, "02_features.png", config["plot_dpi"])


# --------------------------------------------------------------------------- #
# Plot 03 — Visualização do split
# --------------------------------------------------------------------------- #
def plot_dataset_split(df: pd.DataFrame, splits: dict, config: dict, run_dir: str) -> str:
    """Plota proporção dos splits e distribuição da MA em cada um."""
    _apply_style(config)
    train_end, val_end = splits["train_end"], splits["val_end"]
    n = len(df)
    ma = df["ma_target"].values
    parts = {
        "Treino": (ma[:train_end], "tab:blue"),
        "Val": (ma[train_end:val_end], "tab:green"),
        "Test": (ma[val_end:], "tab:red"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    left = 0
    for nome, (vals, c) in parts.items():
        frac = len(vals) / n if n else 0
        ax1.barh(0, frac, left=left, color=c, label=f"{nome}: {len(vals)}")
        ax1.text(left + frac / 2, 0, f"{frac*100:.0f}%", ha="center", va="center", color="white")
        left += frac
    ax1.set_xlim(0, 1)
    ax1.set_yticks([])
    ax1.set_title("Proporção dos splits (nº de linhas)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8)

    for nome, (vals, c) in parts.items():
        if len(vals):
            ax2.hist(vals, bins=40, alpha=0.5, color=c, density=True, label=nome)
    ax2.set_title("Distribuição da MA por split")
    ax2.legend(fontsize=8)

    means = [np.mean(v) for v, _ in parts.values() if len(v)]
    aviso = ""
    if means and (max(means) - min(means)) / (np.mean(means) + 1e-12) > 0.1:
        aviso = " [AVISO] distribuicoes divergentes entre splits"
    fig.suptitle(f"Split do dataset — {_run_id(config)}{aviso}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, run_dir, "03_dataset_split.png", config["plot_dpi"])


# --------------------------------------------------------------------------- #
# Plot 04 — Curvas de loss + diagnóstico
# --------------------------------------------------------------------------- #
def plot_training_loss(loss_history: list, config: dict, run_dir: str) -> str:
    """Plota curvas de loss (log) com painel textual de diagnóstico."""
    _apply_style(config)
    diag = diagnose_learning(loss_history)
    epochs = [h["epoch"] for h in loss_history]
    train = [h["train_loss"] for h in loss_history]
    val = [h["val_loss"] for h in loss_history]

    fig, (ax, ax_txt) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})

    ax.plot(epochs, train, color="tab:blue", label="train_loss")
    ax.plot(epochs, val, color="tab:orange", label="val_loss")
    ax.set_yscale("log")
    best_epoch = diag["best_epoch"]
    ax.axvline(best_epoch, color="green", ls=":", lw=1.5, label=f"melhor (ép. {best_epoch})")
    ax.annotate(
        f"min val={diag['best_val_loss']:.6f}",
        xy=(best_epoch, diag["best_val_loss"]),
        xytext=(0.55, 0.85), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="green"), fontsize=9,
    )
    # Sombra após early stopping (se parou antes do total de épocas)
    if epochs and epochs[-1] < config["epochs"]:
        ax.axvspan(epochs[-1], config["epochs"], color="gray", alpha=0.15)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (log)")
    ax.set_title(f"Curvas de loss — {_run_id(config)}")
    ax.legend(fontsize=8)

    checks = "\n".join(f"   {c}" for c in diag["checks"])
    texto = (
        "DIAGNOSTICO DE APRENDIZADO\n"
        "─────────────────────────────\n"
        f"Epocas treinadas   : {diag['epochs_trained']} / {config['epochs']}\n"
        f"Melhor val_loss    : {diag['best_val_loss']:.6f} (epoca {best_epoch})\n"
        f"Loss final treino  : {diag['final_train_loss']:.6f}\n"
        f"Loss final val     : {diag['final_val_loss']:.6f}\n"
        f"Gap treino->val    : {diag['gap_pct']:+.1f}%\n\n"
        f"Padrao: {diag['titulo']}\n"
        f"{checks}\n\n"
        f"Sugestao: {diag['sugestao']}\n"
        "─────────────────────────────"
    )
    ax_txt.axis("off")
    ax_txt.text(
        0.0, 1.0, texto, fontsize=9, family="monospace", va="top", ha="left",
        transform=ax_txt.transAxes,
    )
    fig.tight_layout()
    return _save(fig, run_dir, "04_training_loss.png", config["plot_dpi"])


# --------------------------------------------------------------------------- #
# Plot 05 — MA real vs prevista no test set
# --------------------------------------------------------------------------- #
def plot_predictions_test(preds: dict, metrics: dict, config: dict, run_dir: str) -> str:
    """Plota MA real vs previsão h=1 e o envelope até h=H, com erro absoluto."""
    _apply_style(config)
    y_true = preds["y_true_price"]
    y_pred = preds["y_pred_price"]
    t = pd.to_datetime(preds["anchor_time"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(t, y_true[:, 0], color="tab:blue", lw=1.2, label="MA real")
    ax1.plot(t, y_pred[:, 0], color="tab:orange", ls="--", lw=1.0, label="Previsto h=1")
    env_low = np.min(y_pred, axis=1)
    env_high = np.max(y_pred, axis=1)
    ax1.fill_between(t, env_low, env_high, color="tab:orange", alpha=0.2, label="Envelope h=1..H")

    g = metrics["global"]
    ax1.set_title(
        f"MA real vs prevista — {_run_id(config)} | "
        f"MAE={g['mae_pips']} pips | Dir={g['directional_acc']*100:.1f}%"
    )
    ax1.set_ylabel("MA")
    ax1.legend(fontsize=8)

    err = np.abs(y_pred[:, 0] - y_true[:, 0])
    ax2.plot(t, err, color="tab:red", lw=0.8)
    ax2.set_ylabel("|erro| h=1")
    ax2.set_xlabel("Tempo")
    fig.tight_layout()
    return _save(fig, run_dir, "05_predictions_test.png", config["plot_dpi"])


# --------------------------------------------------------------------------- #
# Plot 06 — Distribuição dos erros
# --------------------------------------------------------------------------- #
def plot_error_distribution(preds: dict, config: dict, run_dir: str) -> str:
    """Histograma + KDE dos erros (h=1) em pips, com anotação de viés."""
    _apply_style(config)
    from evaluate.metrics import _pip_size

    pip = _pip_size(config["symbol"])
    err = (preds["y_pred_price"][:, 0] - preds["y_true_price"][:, 0]) / pip

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(err, bins=60, color="tab:blue", alpha=0.6, density=True)

    # KDE simples (gaussiano) sem dependências extras
    if len(err) > 1 and np.std(err) > 0:
        xs = np.linspace(err.min(), err.max(), 200)
        bw = 1.06 * np.std(err) * len(err) ** (-1 / 5)
        kde = np.mean(
            np.exp(-0.5 * ((xs[:, None] - err[None, :]) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)),
            axis=1,
        )
        ax.plot(xs, kde, color="tab:red", lw=1.5, label="KDE")

    ax.axvline(0.0, color="black", ls="--", lw=1)
    media = float(np.mean(err))
    std = float(np.std(err))
    pos_pct = float(np.mean(err > 0) * 100)
    ax.set_title(f"Distribuição dos erros (pips) — {_run_id(config)}")
    ax.set_xlabel("Erro (pips)")
    txt = f"média={media:.2f} | σ={std:.2f} | %positivos={pos_pct:.1f}%"
    if abs(media) > 2:
        sentido = "superestimar" if media > 0 else "subestimar"
        txt += f"\nAVISO: Vies detectado - modelo tende a {sentido}"
    ax.text(
        0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, run_dir, "06_error_distribution.png", config["plot_dpi"])


# --------------------------------------------------------------------------- #
# Plot 07 — Acurácia direcional por horizonte
# --------------------------------------------------------------------------- #
def plot_directional_accuracy(metrics: dict, config: dict, run_dir: str) -> str:
    """Barras horizontais da acurácia direcional por horizonte."""
    _apply_style(config)
    by_h = metrics["by_horizon"]
    hs = [f"h={b['h']}" for b in by_h]
    accs = [b["directional_acc"] * 100 for b in by_h]

    def color(a: float) -> str:
        if a > 55:
            return "tab:green"
        if a >= 50:
            return "gold"
        return "tab:red"

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(hs, accs, color=[color(a) for a in accs])
    ax.axvline(50, color="red", ls="--", lw=1.2, label="baseline 50%")
    for bar, a in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{a:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Acurácia direcional (%)")
    ax.set_title(f"Acurácia Direcional — % de acerto na direção da MA\n{_run_id(config)}")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, run_dir, "07_directional_accuracy.png", config["plot_dpi"])


# --------------------------------------------------------------------------- #
# Plot 08 — Backtest overlay
# --------------------------------------------------------------------------- #
def plot_backtest_overlay(preds: dict, metrics: dict, config: dict, run_dir: str) -> str:
    """Overlay da MA real vs prevista (h=1) nas últimas N barras de teste."""
    _apply_style(config)
    from evaluate.metrics import _pip_size

    pip = _pip_size(config["symbol"])
    n_bars = min(config["backtest_bars"], len(preds["anchor_time"]))
    sl = slice(-n_bars, None)

    t = pd.Series(pd.to_datetime(preds["anchor_time"][sl])).reset_index(drop=True)
    ma_real = preds["y_true_price"][sl, 0]
    ma_pred = preds["y_pred_price"][sl, 0]
    anchor_ma = preds["anchor_ma"][sl]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    ax1.plot(t, ma_real, color="tab:blue", lw=1.6, label="MA real")
    ax1.plot(t, ma_pred, color="tab:orange", ls="--", lw=1.0, label="MA prevista (NN)")
    ax1.fill_between(t, ma_real, ma_pred, where=ma_pred >= ma_real, color="tab:green", alpha=0.25, label="NN otimista")
    ax1.fill_between(t, ma_real, ma_pred, where=ma_pred < ma_real, color="tab:red", alpha=0.25, label="NN pessimista")
    ax1.set_ylabel("MA")
    ax1.legend(fontsize=8, loc="best")

    err_pips = np.abs(ma_real - ma_pred) / pip
    mae = metrics["global"]["mae_pips"]
    ax2.bar(t, err_pips, width=0.02, color=["tab:green" if e <= mae else "tab:red" for e in err_pips])
    ax2.axhline(mae, color="black", ls="--", lw=1, label=f"MAE global ({mae})")
    ax2.set_ylabel("|erro| pips")
    ax2.legend(fontsize=8)

    true_dir = np.sign(ma_real - anchor_ma)
    pred_dir = np.sign(ma_pred - anchor_ma)
    acertos = (true_dir == pred_dir).astype(float)
    cum_acc = np.cumsum(acertos) / np.arange(1, len(acertos) + 1) * 100
    ax3.plot(t, cum_acc, color="tab:purple", lw=1.2)
    ax3.axhline(50, color="red", ls="--", lw=1)
    ax3.axhline(55, color="green", ls="--", lw=1)
    if len(cum_acc):
        ax3.annotate(f"{cum_acc[-1]:.1f}%", xy=(t.iloc[-1], cum_acc[-1]), fontsize=9, color="tab:purple")
    ax3.set_ylabel("Dir. acc acum. (%)")
    ax3.set_xlabel("Tempo")

    ini = t.iloc[0].date() if len(t) else "?"
    fim = t.iloc[-1].date() if len(t) else "?"
    g = metrics["global"]
    fig.suptitle(
        f"{config['symbol']} {config['timeframe']} | SMA {config['ma_period']} | "
        f"NN Forecast +{config['forecast_steps']} barras\n"
        f"Período: {ini} a {fim} | MAE: {g['mae_pips']} pips | "
        f"Dir. Acc: {g['directional_acc']*100:.1f}%",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, run_dir, "08_backtest_overlay.png", config["plot_dpi"])


# --------------------------------------------------------------------------- #
# Orquestradores
# --------------------------------------------------------------------------- #
def generate_diagnostic_plots(
    df: pd.DataFrame,
    splits: dict,
    loss_history: list,
    preds: dict,
    metrics: dict,
    config: dict,
    run_dir: str,
) -> None:
    """Gera os plots 01 a 07 (diagnóstico)."""
    plot_raw_price(df, splits, config, run_dir)
    plot_features(df, config, run_dir)
    plot_dataset_split(df, splits, config, run_dir)
    plot_training_loss(loss_history, config, run_dir)
    plot_predictions_test(preds, metrics, config, run_dir)
    plot_error_distribution(preds, config, run_dir)
    plot_directional_accuracy(metrics, config, run_dir)


def generate_backtest_plot(preds: dict, metrics: dict, config: dict, run_dir: str) -> None:
    """Gera o plot 08 (backtest overlay)."""
    plot_backtest_overlay(preds, metrics, config, run_dir)
