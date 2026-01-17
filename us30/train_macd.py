import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import f1_score, classification_report, confusion_matrix


# =========================
# Config
# =========================

@dataclass
class TrainConfig:
    csv_path: str = "out/us30_h1_macd_dataset.csv"
    out_dir: str = "train_out"

    # Walk-forward split (por tempo)
    train_frac: float = 0.70
    val_frac: float = 0.15  # restante vira test (0.15)

    # Treino
    seed: int = 42
    epochs: int = 30
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 8  # early stop na val macro-F1

    # Modelo
    hidden: int = 128
    dropout: float = 0.20

    # ONNX
    onnx_path: str = "train_out/us30_macd_mlp.onnx"


FEATURE_COLS = (
    [f"ret_{k}" for k in range(1, 8)]
    + ["ret_sum_3", "ret_vol_7"]
    + ["macd_hist", "macd_hist_slope_1", "macd_hist_slope_3"]
    + ["ema50_slope_5", "dist_close_ema50"]
    + ["adx14", "atr_pct", "bb_width_20", "macd_compress"]
)

LABEL_COL = "label"


# =========================
# Utils
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


class StandardScalerNP:
    """
    StandardScaler simples (numpy) com fit apenas no treino.
    """
    def __init__(self, eps=1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)


# =========================
# Dataset
# =========================

class CSVDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# =========================
# Model
# =========================

class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)  # 2 classes: REV(0), CONT(1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Train / Eval
# =========================

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_y = []
    all_p = []
    total_loss = 0.0
    crit = nn.CrossEntropyLoss()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = crit(logits, y)
        total_loss += loss.item() * len(y)

        pred = torch.argmax(logits, dim=1)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_p) if all_p else np.array([])

    if len(y_true) == 0:
        return {"loss": np.nan, "acc": np.nan, "macro_f1": np.nan}, y_true, y_pred

    loss_avg = total_loss / len(y_true)
    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {"loss": loss_avg, "acc": acc, "macro_f1": macro_f1}, y_true, y_pred


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ---------- Load ----------
    df = pd.read_csv(cfg.csv_path)

    # Se seu CSV salvou index time, normalmente vem uma coluna tipo "time" ou "Unnamed: 0"
    # Tentamos preservar a ordem temporal sem depender dessa coluna.
    # Apenas garantimos que não foi embaralhado.
    for c in ["time", "datetime", "date"]:
        if c in df.columns:
            # se existir, ordena por ela
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.sort_values(c)
            break

    # Mantém só colunas necessárias
    df = df[FEATURE_COLS + [LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    train_df, val_df, test_df = time_split(df, cfg.train_frac, cfg.val_frac)

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df[LABEL_COL].to_numpy(dtype=np.int64)

    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL_COL].to_numpy(dtype=np.int64)

    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COL].to_numpy(dtype=np.int64)

    # ---------- Scale (fit no treino) ----------
    scaler = StandardScalerNP()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # salva scaler (pra usar no MQL5 depois)
    scaler_path = os.path.join(cfg.out_dir, "scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(
            {"mean": scaler.mean_.tolist(), "std": scaler.std_.tolist(), "features": FEATURE_COLS},
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---------- Sampler balanceado ----------
    # pesos inversamente proporcionais à frequência
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_ds = CSVDataset(X_train, y_train)
    val_ds = CSVDataset(X_val, y_val)
    test_ds = CSVDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(n_in=len(FEATURE_COLS), hidden=cfg.hidden, dropout=cfg.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "best_model.pt")
    patience_left = cfg.patience

    history = []

    # ---------- Train loop ----------
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = crit(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            n_seen += len(yb)

        train_loss = total_loss / max(n_seen, 1)

        val_metrics, _, _ = eval_model(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_macroF1={val_metrics['macro_f1']:.4f}"
        )

        # early stop por macro-f1
        if val_metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # salva histórico
    pd.DataFrame(history).to_csv(os.path.join(cfg.out_dir, "history.csv"), index=False)

    # ---------- Test ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, y_true, y_pred = eval_model(model, test_loader, device)

    print("\n=== TEST ===")
    print(f"loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} macroF1={test_metrics['macro_f1']:.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nReport:\n", classification_report(y_true, y_pred, digits=4))

    # ---------- Export ONNX ----------
    model.eval()
    model_cpu = model.to("cpu")

    dummy = torch.randn(1, len(FEATURE_COLS), dtype=torch.float32)
    onnx_path = cfg.onnx_path
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
        keep_initializers_as_inputs=False,
        export_params=True,
        external_data=False
    )

    print(f"\nSaved ONNX: {onnx_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved best: {best_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)