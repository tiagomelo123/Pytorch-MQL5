import os
import json
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    csv_path: str = BASE_DIR /  "out/us30_h1_ma_delta_h10.csv"

    # columns
    time_col: str = "time"
    feature_cols: tuple = ("ret_1", "hl", "body", "dist_ema", "slope_ema", "atr_norm")
    target_col: str = "y_ema_delta_h10"   # 1 alvo (regressão)

    # windowing (CNN1D)
    lookback: int = 64   # N velas
    horizon: int = 10    # só documental; label já está no CSV

    # training
    batch_size: int = 256
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # splits (temporal)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 0.15 (resto)

    # outputs
    out_dir: str = BASE_DIR / "train_out"
    scaler_path: str = BASE_DIR /  "train_out/scaler.json"
    best_pt_path: str = BASE_DIR /  "train_out/cnn1d_best.pt"
    onnx_path: str = BASE_DIR / "train_out/us30_h1_cnn1d_ma_delta_h10.onnx"

cfg = CFG()


# =========================
# UTILS
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_scaler_json(path: str, mean: np.ndarray, std: np.ndarray, feature_cols):
    payload = {
        "type": "standard",
        "feature_cols": list(feature_cols),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "eps": 1e-8
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_csv(csv_path: str, time_col: str):
    df = pd.read_csv(csv_path)
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    return df


# =========================
# WINDOWED DATASET
# =========================
class WindowDataset(Dataset):
    """
    Retorna:
      X: (F, L)  float32   [CNN1D: canais=features, comprimento=lookback]
      y: (1,)    float32
    """
    def __init__(self, X_2d: np.ndarray, y_1d: np.ndarray, lookback: int):
        assert X_2d.ndim == 2
        assert y_1d.ndim == 1
        self.X = X_2d
        self.y = y_1d
        self.lookback = lookback

        self.n = len(self.y)
        self.start = lookback - 1  # primeiro índice que tem janela completa
        self.length = self.n - self.start
        if self.length <= 0:
            raise ValueError("Poucos dados para o lookback escolhido.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t = self.start + idx
        window = self.X[t - self.lookback + 1 : t + 1]  # (L, F)
        # transpor p/ (F, L) para Conv1d
        X = window.T.astype(np.float32)                 # (F, L)
        y = np.array([self.y[t]], dtype=np.float32)     # (1,)
        return X, y


# =========================
# MODEL: CNN1D
# =========================
class CNN1DRegressor(nn.Module):
    def __init__(self, n_features: int, lookback: int):
        super().__init__()
        # Entrada: (B, C=n_features, L=lookback)
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.AdaptiveAvgPool1d(1),  # (B, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),            # (B, 64)
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1)         # regressão
        )

    def forward(self, x):
        z = self.net(x)
        y = self.head(z)
        return y


# =========================
# TRAIN LOOP
# =========================
@torch.no_grad()
def eval_loop(model, loader, loss_fn, device):
    model.eval()
    losses = []
    maes = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        maes.append(torch.mean(torch.abs(pred - y)).item())
    return float(np.mean(losses)), float(np.mean(maes))

def train_loop(model, train_loader, val_loader, device, epochs, lr, weight_decay, out_best_path):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber (SmoothL1)
    best_val = float("inf")
    best_epoch = -1
    patience = 10
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        tr_maes = []

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_losses.append(loss.item())
            tr_maes.append(torch.mean(torch.abs(pred - y)).item())

        tr_loss = float(np.mean(tr_losses))
        tr_mae = float(np.mean(tr_maes))
        val_loss, val_mae = eval_loop(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.6f} train_mae={tr_mae:.6f} | val_loss={val_loss:.6f} val_mae={val_mae:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict()}, out_best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (best epoch={best_epoch}, best val_loss={best_val:.6f})")
                break

    print(f"Best: epoch={best_epoch}, val_loss={best_val:.6f}")
    return best_epoch, best_val


# =========================
# ONNX EXPORT
# =========================
def export_onnx(model, onnx_path, n_features, lookback, device):
    model.eval()
    dummy = torch.zeros(1, n_features, lookback, dtype=torch.float32, device=device)
    # Exporta com batch dinâmico
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["X"],
        output_names=["yhat"],        
        opset_version=18,
        dynamic_axes=None,
        keep_initializers_as_inputs=False,
        export_params=True,
        external_data=False
    )
    print(f"OK: ONNX salvo em {onnx_path}")


# =========================
# MAIN
# =========================
def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    df = load_csv(cfg.csv_path, cfg.time_col)

    missing = [c for c in (list(cfg.feature_cols) + [cfg.target_col]) if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colunas faltando no CSV: {missing}")

    X = df[list(cfg.feature_cols)].astype(np.float32).to_numpy()
    y = df[cfg.target_col].astype(np.float32).to_numpy()

    # Split temporal por índice (antes do windowing, mas ok porque o Dataset só usa passado)
    n = len(df)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * (cfg.train_frac + cfg.val_frac))
    # train: [0:n_train), val: [n_train:n_val), test: [n_val:n)

    # Fit scaler apenas no treino
    mean = X[:n_train].mean(axis=0)
    std = X[:n_train].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    Xs = (X - mean) / std
    save_scaler_json(cfg.scaler_path, mean, std, cfg.feature_cols)
    print(f"OK: scaler salvo em {cfg.scaler_path}")

    # Datasets windowed (cada split precisa ter dados suficientes para lookback)
    train_ds = WindowDataset(Xs[:n_train], y[:n_train], cfg.lookback)
    val_ds   = WindowDataset(Xs[n_train:n_val], y[n_train:n_val], cfg.lookback)
    test_ds  = WindowDataset(Xs[n_val:], y[n_val:], cfg.lookback)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=False)

    n_features = len(cfg.feature_cols)
    model = CNN1DRegressor(n_features=n_features, lookback=cfg.lookback).to(cfg.device)

    print(f"Device: {cfg.device} | rows={n} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    best_epoch, best_val = train_loop(
        model, train_loader, val_loader,
        device=cfg.device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        out_best_path=cfg.best_pt_path
    )

    # Carrega melhor e avalia no teste
    ckpt = torch.load(cfg.best_pt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])

    loss_fn = nn.SmoothL1Loss(beta=1.0)
    test_loss, test_mae = eval_loop(model, test_loader, loss_fn, cfg.device)
    print(f"TEST | loss={test_loss:.6f} mae={test_mae:.6f}")

    # Export ONNX
    export_onnx(model, cfg.onnx_path, n_features, cfg.lookback, cfg.device)

if __name__ == "__main__":
    main()
