"""Loop de treino com early stopping, adaptado para reportar progresso ao Streamlit.

Suporta três modos de saída (``output_mode``):

- ``"regressao"``: ``MSELoss``, saída escalar, métrica extra = MAE.
- ``"classificacao_binaria"``: ``BCEWithLogitsLoss`` (com ``pos_weight``
  opcional), saída escalar (logit), métrica extra = acurácia.
- ``"classificacao_multiclasse"``: ``CrossEntropyLoss`` (com ``class_weights``
  opcional), saída ``(batch, n_classes)``, métrica extra = acurácia.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

ProgressCallback = Callable[[int, int, float, float, float, float], None]

MODOS_VALIDOS = ("regressao", "classificacao_binaria", "classificacao_multiclasse")


def get_device() -> torch.device:
    """Detecta o melhor device disponível: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Fixa as seeds de torch/numpy para reprodutibilidade."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_criterion(output_mode: str, pos_weight: float | None, class_weights: list[float] | None, device: torch.device) -> nn.Module:
    if output_mode == "regressao":
        return nn.MSELoss()
    if output_mode == "classificacao_binaria":
        pw = torch.tensor(pos_weight, device=device) if pos_weight else None
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if output_mode == "classificacao_multiclasse":
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
        return nn.CrossEntropyLoss(weight=cw)
    raise ValueError(f"output_mode inválido: {output_mode}. Use um de {MODOS_VALIDOS}.")


def _batch_metric(output_mode: str, pred: torch.Tensor, y: torch.Tensor) -> float:
    """Métrica extra (além da loss) calculada por batch: acurácia ou MAE."""
    if output_mode == "regressao":
        return torch.abs(pred - y).mean().item()
    if output_mode == "classificacao_binaria":
        preds = (torch.sigmoid(pred) > 0.5).float()
        return (preds == y).float().mean().item()
    # classificacao_multiclasse
    preds = torch.argmax(pred, dim=1)
    return (preds == y).float().mean().item()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    output_mode: str,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    """Executa uma época de treino (se ``optimizer``) ou avaliação.

    Returns:
        Tupla ``(loss_medio, metrica_extra)``.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_metric, total_n = 0.0, 0.0, 0
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_metric += _batch_metric(output_mode, pred, y) * bs
            total_n += bs

    return total_loss / max(1, total_n), total_metric / max(1, total_n)


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    config: dict,
    output_mode: str,
    progress_callback: ProgressCallback | None = None,
    pos_weight: float | None = None,
    class_weights: list[float] | None = None,
    stop_event: "threading.Event | None" = None,
) -> dict:
    """Treina o modelo com early stopping.

    Args:
        model: modelo a treinar.
        loaders: dicionário com DataLoaders ``"train"`` e ``"val"``.
        config: dicionário com ``epochs, learning_rate, weight_decay,
            patience, seed``.
        output_mode: um de ``"regressao"``, ``"classificacao_binaria"``,
            ``"classificacao_multiclasse"``.
        progress_callback: função opcional chamada a cada época com
            ``(epoch, total_epochs, train_loss, val_loss, train_metric,
            val_metric)`` — usada pela página do Streamlit para atualizar a
            barra de progresso e o gráfico ao vivo.
        pos_weight: peso da classe positiva na ``BCEWithLogitsLoss``
            (só ``classificacao_binaria``).
        class_weights: pesos por classe na ``CrossEntropyLoss``
            (só ``classificacao_multiclasse``).
        stop_event: ``threading.Event`` opcional — quando setado (checado no
            início de cada época), interrompe o treino antes de terminar
            todas as épocas, mantendo o melhor checkpoint já encontrado até
            então. Permite um botão "parar treinamento" na UI, rodando o
            treino em uma thread separada.

    Returns:
        Dicionário com ``loss_history`` (lista de dicts), ``best_epoch``,
        ``best_state`` (state_dict do melhor modelo), ``device`` e
        ``interrompido`` (``True`` se parado manualmente via ``stop_event``
        antes de terminar as épocas ou o early stopping por paciência).
    """
    set_seed(config["seed"])
    device = get_device()
    model = model.to(device)

    criterion = _build_criterion(output_mode, pos_weight, class_weights, device)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    loss_history: list[dict] = []

    epochs = config["epochs"]
    patience = config["patience"]
    interrompido = False

    for epoch in range(1, epochs + 1):
        if stop_event is not None and stop_event.is_set():
            interrompido = True
            break
        t0 = time.time()
        train_loss, train_metric = _run_epoch(model, loaders["train"], criterion, device, output_mode, optimizer)
        val_loss, val_metric = _run_epoch(model, loaders["val"], criterion, device, output_mode)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        dt_epoch = time.time() - t0

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        loss_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metric": train_metric,
                "val_metric": val_metric,
                "lr": lr,
                "segundos": dt_epoch,
            }
        )

        if progress_callback is not None:
            progress_callback(epoch, epochs, train_loss, val_loss, train_metric, val_metric)

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return {
        "loss_history": loss_history,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "device": str(device),
        "interrompido": interrompido,
    }


def evaluate(model: nn.Module, loader: DataLoader, output_mode: str) -> dict:
    """Avalia o modelo em um conjunto (ex.: teste) e retorna métricas finais."""
    device = get_device()
    model = model.to(device)
    model.eval()

    preds_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            preds_all.append(pred.cpu())
            y_all.append(y)
    preds = torch.cat(preds_all)
    y = torch.cat(y_all)

    if output_mode == "classificacao_binaria":
        probs = torch.sigmoid(preds)
        pred_labels = (probs > 0.5).float()
        acc = (pred_labels == y).float().mean().item()
        tp = ((pred_labels == 1) & (y == 1)).sum().item()
        fp = ((pred_labels == 1) & (y == 0)).sum().item()
        fn = ((pred_labels == 0) & (y == 1)).sum().item()
        precisao = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precisao * recall / max(1e-9, precisao + recall)
        return {
            "acuracia": acc,
            "precisao": precisao,
            "recall": recall,
            "f1": f1,
            "predicoes": pred_labels.numpy(),
            "reais": y.numpy(),
            "probabilidades": probs.numpy(),
        }

    if output_mode == "classificacao_multiclasse":
        probs = torch.softmax(preds, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        y_long = y.long()
        n_classes = preds.shape[1]
        acc = (pred_labels == y_long).float().mean().item()

        matriz_confusao = torch.zeros((n_classes, n_classes), dtype=torch.int64)
        for real, previsto in zip(y_long.tolist(), pred_labels.tolist()):
            matriz_confusao[real, previsto] += 1

        f1s = []
        for c in range(n_classes):
            tp = ((pred_labels == c) & (y_long == c)).sum().item()
            fp = ((pred_labels == c) & (y_long != c)).sum().item()
            fn = ((pred_labels != c) & (y_long == c)).sum().item()
            prec_c = tp / max(1, tp + fp)
            rec_c = tp / max(1, tp + fn)
            f1s.append(2 * prec_c * rec_c / max(1e-9, prec_c + rec_c))
        f1_macro = float(np.mean(f1s))

        return {
            "acuracia": acc,
            "f1_macro": f1_macro,
            "f1_por_classe": f1s,
            "matriz_confusao": matriz_confusao.numpy(),
            "predicoes": pred_labels.numpy(),
            "reais": y_long.numpy(),
            "probabilidades": probs.numpy(),
        }

    # regressao
    mae = torch.abs(preds - y).mean().item()
    rmse = torch.sqrt(torch.mean((preds - y) ** 2)).item()
    ss_res = torch.sum((y - preds) ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predicoes": preds.numpy(),
        "reais": y.numpy(),
    }
