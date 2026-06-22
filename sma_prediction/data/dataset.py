"""Dataset PyTorch, sliding window, split temporal e normalização sem leakage."""

import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from data.features import get_feature_columns, get_target_columns

logger = logging.getLogger(__name__)


class SlidingWindowDataset(Dataset):
    """Dataset de janelas deslizantes para previsão Seq2Seq da MA.

    Cada amostra é composta por uma janela de ``lookback`` barras de features e
    um target multi-step com ``forecast_steps`` deltas normalizados por ATR.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        anchor_ma: np.ndarray,
        anchor_atr: np.ndarray,
        anchor_time: np.ndarray,
        lookback: int,
    ) -> None:
        """Inicializa o dataset.

        Args:
            features: matriz normalizada ``(n_rows, num_features)``.
            targets: matriz de deltas ``(n_rows, forecast_steps)``.
            anchor_ma: MA real na barra-âncora de cada linha ``(n_rows,)``.
            anchor_atr: ATR real na barra-âncora de cada linha ``(n_rows,)``.
            anchor_time: timestamp da barra-âncora ``(n_rows,)``.
            lookback: tamanho da janela de entrada.
        """
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.anchor_ma = anchor_ma
        self.anchor_atr = anchor_atr
        self.anchor_time = anchor_time
        self.lookback = lookback
        # Índice válido: precisa de 'lookback' barras de contexto antes da âncora
        self.n_windows = len(features) - lookback + 1

    def __len__(self) -> int:
        return max(0, self.n_windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna ``(X, y)`` da janela ``idx``.

        A barra-âncora (onde o target é ancorado) é ``idx + lookback - 1``.
        """
        x = self.features[idx : idx + self.lookback]
        anchor = idx + self.lookback - 1
        y = self.targets[anchor]
        return torch.from_numpy(x), torch.from_numpy(y)

    def anchor_arrays(self) -> dict:
        """Retorna ma/atr/time/target das âncoras de cada janela (alinhados)."""
        start = self.lookback - 1
        return {
            "ma": self.anchor_ma[start:],
            "atr": self.anchor_atr[start:],
            "time": self.anchor_time[start:],
            "targets": self.targets[start:],
        }


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """Calcula os índices de corte temporal para treino/val/test.

    Args:
        n: número total de linhas.
        train_ratio: fração de treino.
        val_ratio: fração de validação.

    Returns:
        Tupla ``(train_end, val_end)`` com índices de corte.
    """
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def build_datasets(df_features: pd.DataFrame, config: dict, run_dir: str) -> dict:
    """Cria janelas, splits temporais, scaler (sem leakage) e DataLoaders.

    O ``MinMaxScaler`` é fitado **apenas** nas linhas de treino e aplicado via
    ``transform`` em val/test. O target (delta/ATR) é adimensional e não usa
    scaler. O scaler é salvo em ``run_dir/scaler.pkl``.

    Args:
        df_features: DataFrame retornado por ``build_features``.
        config: dicionário de configuração.
        run_dir: diretório de artefatos do run atual.

    Returns:
        Dicionário com loaders, datasets, scaler, colunas e metadados de split.
    """
    timeframe = config["timeframe"]
    lookback = config["lookback_window"]
    horizon = config["forecast_steps"]
    batch_size = config["batch_size"]

    feature_cols = get_feature_columns(timeframe)
    target_cols = get_target_columns(horizon)
    num_features = len(feature_cols)

    n = len(df_features)
    train_end, val_end = _split_indices(n, config["train_ratio"], config["val_ratio"])

    # --- Scaler fitado SOMENTE no treino (sem data leakage) ---
    scaler = MinMaxScaler()
    scaler.fit(df_features[feature_cols].iloc[:train_end])

    feat_all = scaler.transform(df_features[feature_cols])
    targ_all = df_features[target_cols].values
    ma_all = df_features["ma"].values
    atr_all = df_features["atr"].values
    time_all = df_features["time"].values

    os.makedirs(run_dir, exist_ok=True)
    scaler_path = os.path.join(run_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Cada split inclui 'lookback-1' barras de contexto do split anterior para
    # que as janelas não fiquem vazias nas bordas, mantendo o corte temporal das
    # âncoras (a âncora de cada janela pertence ao split correto).
    def make_subset(start: int, end: int) -> SlidingWindowDataset:
        ctx = max(0, start - (lookback - 1))
        return SlidingWindowDataset(
            feat_all[ctx:end],
            targ_all[ctx:end],
            ma_all[ctx:end],
            atr_all[ctx:end],
            time_all[ctx:end],
            lookback,
        )

    train_ds = make_subset(0, train_end)
    val_ds = make_subset(train_end, val_end)
    test_ds = make_subset(val_end, n)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def fmt(ds: SlidingWindowDataset) -> str:
        t = ds.anchor_arrays()["time"]
        if len(t) == 0:
            return "vazio"
        ini = pd.Timestamp(t[0]).strftime("%Y-%m-%d")
        fim = pd.Timestamp(t[-1]).strftime("%Y-%m-%d")
        return f"{len(ds)} janelas ({ini} a {fim})"

    total = len(train_ds) + len(val_ds) + len(test_ds)
    logger.info("Janelas: %d total", total)
    logger.info("   Treino : %s", fmt(train_ds))
    logger.info("   Val    : %s", fmt(val_ds))
    logger.info("   Test   : %s", fmt(test_ds))
    logger.info("Scaler salvo: %s", scaler_path)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "num_features": num_features,
        "train_end": train_end,
        "val_end": val_end,
        "n_rows": n,
    }
