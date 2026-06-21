"""Loop de treino com early stopping, scheduler e grad clipping."""

import copy
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.lstm_seq2seq import LSTMSeq2Seq

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detecta o melhor device disponível: cuda > mps > cpu.

    Returns:
        ``torch.device`` selecionado.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Fixa as seeds de torch e numpy para reprodutibilidade.

    Args:
        seed: valor da seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: LSTMSeq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam = None,
    teacher_forcing_ratio: float = 0.0,
) -> float:
    """Executa uma época de treino (se ``optimizer``) ou avaliação.

    Args:
        model: modelo Seq2Seq.
        loader: DataLoader da época.
        criterion: função de perda.
        device: device de execução.
        optimizer: otimizador; se ``None``, roda em modo avaliação.
        teacher_forcing_ratio: ratio de teacher forcing (treino apenas).

    Returns:
        Loss médio (ponderado por amostra) da época.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_n = 0
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad()
                pred = model(x, target=y, teacher_forcing_ratio=teacher_forcing_ratio)
            else:
                pred = model(x, target=None, teacher_forcing_ratio=0.0)

            loss = criterion(pred, y)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

    return total_loss / max(1, total_n)


def train_model(
    model: LSTMSeq2Seq,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    run_dir: str,
) -> dict:
    """Treina o modelo com early stopping e salva os artefatos.

    Salva ``run_dir/model.pt`` (melhor state_dict) e
    ``run_dir/loss_history.json``.

    Args:
        model: modelo a treinar.
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        config: dicionário de configuração.
        run_dir: diretório de artefatos.

    Returns:
        Dicionário com ``loss_history`` (lista de dicts) e ``best_epoch``.
    """
    set_seed(config["seed"])
    device = get_device()
    model = model.to(device)
    logger.info("Treinando modelo (device: %s)...", device.type)

    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    loss_history: list[dict] = []

    epochs = config["epochs"]
    tf_ratio = config["teacher_forcing_ratio"]
    patience = config["patience"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(
            model, train_loader, criterion, device, optimizer, tf_ratio
        )
        val_loss = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

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
                "lr": lr,
            }
        )

        flag = " 🔥 melhor modelo" if improved else ""
        logger.info(
            "[Época %3d/%d] train_loss=%.6f | val_loss=%.6f | lr=%.6f | ⏱ %.1fs%s",
            epoch,
            epochs,
            train_loss,
            val_loss,
            lr,
            dt,
            flag,
        )

        if epochs_no_improve >= patience:
            logger.info(
                "⏹  Early stopping na época %d (melhor: época %d)", epoch, best_epoch
            )
            break

    # Restaura o melhor modelo
    model.load_state_dict(best_state)

    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(best_state, model_path)
    logger.info("Salvo: %s", model_path)

    history_path = os.path.join(run_dir, "loss_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)

    return {"loss_history": loss_history, "best_epoch": best_epoch, "device": device}
