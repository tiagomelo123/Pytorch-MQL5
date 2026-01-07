# train_pytorch_3class.py
import os, json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# CONFIG (ajuste aqui)
# =========================
CSV_PATH   = "dataset.csv"
OUT_DIR    = "artifacts_3class"

LABEL_COL  = "label"   # coluna com 0/1/2
# Se quiser fixar as features, coloque uma lista. Se None, autodetecta todas numéricas exceto label.
FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
    "atr",
]

SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

TEST_SIZE  = 0.15
VAL_SIZE   = 0.15

BATCH_SIZE = 512
EPOCHS     = 50
LR         = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE   = 8
GRAD_CLIP  = 1.0

HIDDEN     = 128
DROPOUT    = 0.15

LABEL_MAP  = {0: "NEUTRAL", 1: "UP", 2: "RISK"}


# =========================
# Utils
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def get_feature_cols(df: pd.DataFrame, label_col: str) -> list:
    if FEATURE_COLS is not None:
        return FEATURE_COLS
    # autodetect: pega numéricas e remove label
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)
    return num_cols


# =========================
# Dataset
# =========================
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Model
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Train/Eval
# =========================
def run_eval(model, loader, criterion):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss_sum += loss.item() * len(yb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)

    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    set_seed(SEED)
    makedirs(OUT_DIR)

    # 1) Load
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].astype(int)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Não encontrei a coluna LABEL_COL='{LABEL_COL}' no CSV.")

    # garante labels como int 0/1/2
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    feat_cols = get_feature_cols(df, LABEL_COL)
    if len(feat_cols) == 0:
        raise ValueError("Não encontrei colunas numéricas de feature. Defina FEATURE_COLS manualmente.")

    # remove NaNs
    df = df.dropna(subset=feat_cols + [LABEL_COL]).copy()

    X = df[feat_cols].to_numpy()
    y = df[LABEL_COL].to_numpy()

    # 2) Split temporal (shuffle=False)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE), random_state=SEED, shuffle=False
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=SEED, shuffle=False
    )

    # 3) Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # salva scaler
    scaler_payload = {
        "feature_cols": feat_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
    }
    with open(os.path.join(OUT_DIR, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, ensure_ascii=False, indent=2)

    # 4) DataLoaders
    train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TabDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TabDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # 5) Class weights (balanceamento)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = counts.sum() / (counts + 1e-9)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    # 6) Model
    model = MLP(in_dim=X_train.shape[1], hidden=HIDDEN, dropout=DROPOUT, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_path = os.path.join(OUT_DIR, "best_model.pt")
    best_val_loss = float("inf")
    bad_epochs = 0

    # 7) Train loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            loss_sum += loss.item() * len(yb)
            total += len(yb)

        train_loss = loss_sum / max(total, 1)
        val_loss, val_acc = run_eval(model, val_loader, criterion)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # early stopping + save best
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping.")
                break

    # 8) Test report (best)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    target_names = [LABEL_MAP[i] for i in range(3)]
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # 9) Export ONNX
    onnx_path = os.path.join(OUT_DIR, "model_3class.onnx")
    model_cpu = MLP(in_dim=X_train.shape[1], hidden=HIDDEN, dropout=0.0, num_classes=3)
    model_cpu.load_state_dict(torch.load(best_path, map_location="cpu"))
    model_cpu.eval()

    dummy = torch.randn(1, X_train.shape[1], dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"\n✅ Salvo em {OUT_DIR}:")
    print("- best_model.pt")
    print("- model_3class.onnx")
    print("- scaler.json")


if __name__ == "__main__":
    main()
