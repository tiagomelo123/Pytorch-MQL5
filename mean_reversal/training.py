import json
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import onnxruntime as ort


# =========================
# Configurações
# =========================

DATASET_PATH = "mean_reversion_dataset.csv"

MODEL_PT_PATH = "mean_reversion_model.pt"
MODEL_ONNX_PATH = "mean_reversion_model.onnx"
SCALER_PATH = "mean_reversion_scaler.pkl"
METADATA_PATH = "mean_reversion_metadata.json"

THRESHOLD = 0.60
BATCH_SIZE = 32
EPOCHS = 80
LR = 0.001


# =========================
# 1. Carregar dataset
# =========================

df = pd.read_csv(DATASET_PATH)

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

target_col = "label"

drop_cols = [
    "time",
    "open",
    "high",
    "low",
    "close",
    target_col
]

features = [c for c in df.columns if c not in drop_cols]

X = df[features].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

print("Total de linhas:", len(df))
print("Total de features:", len(features))
print("Distribuição do label:")
print(df[target_col].value_counts())


# =========================
# 2. Split temporal
# =========================

split_idx = int(len(df) * 0.8)

X_train = X[:split_idx]
y_train = y[:split_idx]

X_test = X[split_idx:]
y_test = y[split_idx:]


# =========================
# 3. Normalização
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)


# =========================
# 4. DataLoader
# =========================

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


# =========================
# 5. Modelo
# =========================

class MeanReversionNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


model = MeanReversionNN(input_size=X_train_scaled.shape[1])


# =========================
# 6. Loss e otimizador
# =========================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# =========================
# 7. Treinamento
# =========================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")


# =========================
# 8. Avaliação PyTorch
# =========================

model.eval()

with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.sigmoid(logits).numpy().flatten()

print("\nAUC:", roc_auc_score(y_test, probs))

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
    preds = (probs >= threshold).astype(int)

    print("\n======================")
    print("Threshold:", threshold)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))


# =========================
# 8.1 Avaliação por faixas de probabilidade
# =========================

analysis = pd.DataFrame({
    "prob": probs,
    "label": y_test
})

bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
analysis["prob_bin"] = pd.cut(analysis["prob"], bins=bins)

rank_report = analysis.groupby("prob_bin").agg(
    total=("label", "count"),
    winners=("label", "sum"),
    win_rate=("label", "mean"),
    avg_prob=("prob", "mean")
)

print("\nAvaliação por faixa de probabilidade:")
print(rank_report)


# =========================
# 9. Resultado final
# =========================

final_preds = (probs >= THRESHOLD).astype(int)

result = df.iloc[split_idx:].copy()
result["prob_success"] = probs
result["prediction"] = final_preds

print("\nÚltimos sinais:")
print(result[["time", "direction", "label", "prob_success", "prediction"]].tail(20))

# =========================
# 8.2 Salvar erros para análise
# =========================

errors = result[result["label"] != result["prediction"]].copy()
errors.to_csv("mean_reversion_errors.csv", index=False)

high_confidence_errors = result[
    (result["label"] == 0) &
    (result["prob_success"] >= THRESHOLD)
].copy()

high_confidence_errors.to_csv(
    "mean_reversion_high_confidence_errors.csv",
    index=False
)

print("\nErros salvos em mean_reversion_errors.csv")
print("Erros de alta confiança salvos em mean_reversion_high_confidence_errors.csv")


# =========================
# 10. Salvar modelo PyTorch
# =========================

torch.save(model.state_dict(), MODEL_PT_PATH)


# =========================
# 11. Salvar scaler
# =========================

joblib.dump(scaler, SCALER_PATH)


# =========================
# 12. Salvar metadados
# =========================

metadata = {
    "features": features,
    "threshold": THRESHOLD,
    "input_size": len(features),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "drop_cols": drop_cols,
    "target_col": target_col
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nScaler salvo em:", SCALER_PATH)
print("Metadados salvos em:", METADATA_PATH)


# =========================
# 13. Exportar para ONNX
# =========================

dummy_input = torch.randn(1, len(features), dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    MODEL_ONNX_PATH,
    input_names=["input"],
    output_names=["logit"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logit": {0: "batch_size"}
    },
    opset_version=17
)

print("Modelo ONNX salvo em:", MODEL_ONNX_PATH)


# =========================
# 14. Testar ONNX
# =========================

session = ort.InferenceSession(MODEL_ONNX_PATH)

sample = X_test_scaled[0:1].astype(np.float32)

onnx_logit = session.run(
    None,
    {"input": sample}
)[0]

onnx_prob = 1 / (1 + np.exp(-onnx_logit))

print("\nTeste ONNX:")
print("Logit:", onnx_logit)
print("Probabilidade:", onnx_prob)

print("\nProcesso finalizado com sucesso.")