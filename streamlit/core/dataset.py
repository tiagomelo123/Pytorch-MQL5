"""Construção do dataset supervisionado (janelas) e Dataset/DataLoader do PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    """Dataset PyTorch simples para tensores já pré-calculados.

    Args:
        X: janelas de features, shape ``(n, lookback, n_features)``.
        y: alvo. Para regressão/classificação binária, valores contínuos ou
            0/1 (vira ``float32``, usado com MSELoss/BCEWithLogitsLoss).
            Para classificação multi-classe, índices de classe inteiros
            (vira ``int64`` quando ``y_long=True``, usado com CrossEntropyLoss).
        y_long: se ``True``, converte ``y`` para ``torch.long`` (necessário
            para ``nn.CrossEntropyLoss``).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, y_long: bool = False) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long if y_long else torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_target(
    close: pd.Series, horizon: int, tarefa: str
) -> np.ndarray:
    """Constrói o alvo (target) a partir da série de fechamento.

    Args:
        close: série de preços de fechamento.
        horizon: número de barras à frente para o alvo.
        tarefa: uma das chaves de ``config.TAREFAS`` ("Regressão..." ou
            "Classificação...").

    Returns:
        Array 1D alinhado ao índice de ``close`` (últimas ``horizon`` linhas
        ficam com NaN e devem ser descartadas pelo chamador).
    """
    retorno_futuro = close.shift(-horizon) / close - 1
    if tarefa.startswith("Classificação"):
        return (retorno_futuro > 0).astype(float).where(retorno_futuro.notna())
    return retorno_futuro


def build_windows(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    horizon: int,
    tarefa: str,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Monta janelas deslizantes (lookback) e o alvo correspondente.

    Returns:
        Tupla ``(X, y, tempos)`` onde ``X`` tem shape
        ``(n_amostras, lookback, n_features)``, ``y`` tem shape
        ``(n_amostras,)`` e ``tempos`` é o timestamp de referência (última
        barra da janela) de cada amostra.
    """
    target = build_target(features_df["close"], horizon, tarefa)
    valid = target.notna()

    feats = features_df.loc[valid, feature_cols].to_numpy(dtype=np.float32)
    target_arr = target.loc[valid].to_numpy(dtype=np.float32)
    tempos = features_df.loc[valid, "time"].reset_index(drop=True)

    n = len(feats)
    X, y, t_out = [], [], []
    for i in range(lookback, n):
        X.append(feats[i - lookback : i])
        y.append(target_arr[i])
        t_out.append(tempos.iloc[i])

    return np.array(X), np.array(y), pd.Series(t_out)


def build_windows_labeled(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    labels_df: pd.DataFrame,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Monta janelas deslizantes usando rótulos já calculados externamente.

    Usado pela tarefa de pullback/continuação: ``labels_df`` (saída de
    ``labeling.build_pullback_dataset``) traz, por barra, se ela é candidata
    a pullback e o rótulo de continuação (1) ou não (0). Apenas barras
    candidatas viram amostras; as demais barras entram só como contexto nas
    janelas de outras amostras.

    Args:
        features_df: saída de ``features.build_features`` (colunas ``time``
            + features).
        feature_cols: colunas de features a usar.
        labels_df: DataFrame com colunas ``time`` e ``label`` (``NaN`` para
            barras que não são candidatas).
        lookback: tamanho da janela de contexto.

    Returns:
        Tupla ``(X, y, tempos)`` — mesmo formato de ``build_windows``.
    """
    merged = features_df.merge(labels_df[["time", "label"]], on="time", how="left")
    feats = merged[feature_cols].to_numpy(dtype=np.float32)
    labels = merged["label"].to_numpy(dtype=np.float64)
    tempos = merged["time"]

    n = len(feats)
    X, y, t_out = [], [], []
    for i in range(lookback, n):
        if np.isnan(labels[i]):
            continue
        X.append(feats[i - lookback : i])
        y.append(labels[i])
        t_out.append(tempos.iloc[i])

    if not X:
        return np.empty((0, lookback, len(feature_cols)), dtype=np.float32), np.empty(0, dtype=np.float32), pd.Series(dtype="datetime64[ns, UTC]")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.Series(t_out).reset_index(drop=True)


def split_chronological(
    X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float
) -> dict:
    """Divide X/y cronologicamente (sem embaralhar) em treino/validação/teste."""
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        "test": (X[n_train + n_val :], y[n_train + n_val :]),
    }


def scale_splits(splits: dict) -> tuple[dict, StandardScaler]:
    """Normaliza as features com um ``StandardScaler`` ajustado só no treino.

    O scaler é ajustado achatando (amostras*lookback, features) do conjunto
    de treino e aplicado igualmente aos três conjuntos.
    """
    X_train, _ = splits["train"]
    n_feat = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    scaled = {}
    for nome, (X, y) in splits.items():
        shape = X.shape
        X_flat = scaler.transform(X.reshape(-1, n_feat))
        scaled[nome] = (X_flat.reshape(shape), y)
    return scaled, scaler


def make_loaders(splits: dict, batch_size: int, y_long: bool = False) -> dict[str, DataLoader]:
    """Cria DataLoaders de treino (com shuffle) e validação/teste (sem shuffle).

    Args:
        splits: saída de ``scale_splits``/``split_chronological``.
        batch_size: tamanho do batch.
        y_long: repassado a ``WindowDataset`` — use ``True`` para
            classificação multi-classe (``nn.CrossEntropyLoss``).
    """
    loaders = {}
    for nome, (X, y) in splits.items():
        ds = WindowDataset(X, y, y_long=y_long)
        loaders[nome] = DataLoader(
            ds, batch_size=batch_size, shuffle=(nome == "train"), drop_last=False
        )
    return loaders
