# train_pytorch_3class.py
import os, json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# =========================
# CONFIG (ajuste aqui)
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset.csv"

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

def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Segurança: ordena por tempo se existir
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)

    # -------- Retornos (mais importantes) --------
    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_5"]  = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # -------- Range / corpo do candle (normalizados) --------
    df["range_n"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["body_n"]  = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["upper_wick_n"] = (df["high"] - df[["open","close"]].max(axis=1)) / df["close"].replace(0, np.nan)
    df["lower_wick_n"] = (df[["open","close"]].min(axis=1) - df["low"]) / df["close"].replace(0, np.nan)

    # -------- ATR normalizado --------
    if "atr" in df.columns:
        df["atr_n"] = df["atr"] / df["close"].replace(0, np.nan)

    # -------- Médias e distância relativa --------
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    df["dist_ma10"] = (df["close"] - df["ma_10"]) / df["ma_10"]
    df["dist_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["dist_ma50"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    # -------- Inclinação (tendência) --------
    df["ma20_slope"] = df["ma_20"].pct_change(5)

    # -------- Volume relativo (se tick_volume existir) --------
    if "tick_volume" in df.columns:
        vol_ma = df["tick_volume"].rolling(20).mean()
        df["vol_rel"] = df["tick_volume"] / vol_ma

    # -------- Spread relativo (opcional, mas melhor que spread absoluto) --------
    if "spread" in df.columns:
        df["spread_n"] = df["spread"] / df["close"].replace(0, np.nan)

    # Remova colunas intermediárias que você não quer como feature
    # (as MAs podem ser removidas; as distâncias são melhores)
    drop_cols = ["ma_10","ma_20","ma_50"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


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

    # ====== LOAD ======
    df = pd.read_csv(CSV_PATH)
    print("DF rows (raw):", len(df))

    # Se tiver time, garante ordenação temporal
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time").reset_index(drop=True)

    # ====== FEATURE ENGINEERING ======
    df = add_stationary_features(df)

    # Remove real_volume (constante)
    if "real_volume" in df.columns:
        df = df.drop(columns=["real_volume"])

    # Remove OHLC/atr/spread absolutos (vamos usar só as versões normalizadas/relativas)
    drop_abs = ["open", "high", "low", "close", "atr", "spread"]
    df = df.drop(columns=[c for c in drop_abs if c in df.columns])

    # Remove linhas com NaN (por causa de rolling/pct_change)
    df = df.dropna().reset_index(drop=True)
    print("DF rows (after features/dropna):", len(df))

    # ====== LABEL ======
    if "label" not in df.columns:
        raise RuntimeError("Coluna 'label' não encontrada no CSV.")

    y = df["label"].astype(int).to_numpy()

    # ====== FEATURES LIST ======
    FEATURE_COLS = feat_cols = [c for c in df.columns if c not in ("label", "label_name", "time")]
    print("FEATURE_COLS:", FEATURE_COLS)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)

    # Debug: distribuição geral
    print("Class dist ALL:", np.bincount(y, minlength=3))

    # 2) Split temporal (shuffle=False)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE), random_state=SEED, shuffle=False
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=SEED, shuffle=False
    )

    print("Class dist TRAIN:", np.bincount(y_train, minlength=3))
    print("Class dist VAL:  ", np.bincount(y_val, minlength=3))
    print("Class dist TEST: ", np.bincount(y_test, minlength=3))

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

    class_counts = np.bincount(y_train, minlength=3)
    sample_w = 1.0 / class_counts[y_train]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(y_train),
        replacement=True
    )    

    # 4) DataLoaders
    train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler )
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
