"""Diagnóstico automático do padrão de aprendizado a partir do loss history."""

import numpy as np


def diagnose_learning(loss_history: list[dict]) -> dict:
    """Analisa o histórico de loss e retorna um diagnóstico categorizado.

    Verifica, nesta ordem, os padrões: underfitting, overfitting, instável,
    early-stop precoce e, por fim, bom aprendizado (default).

    Args:
        loss_history: lista de dicts ``{epoch, train_loss, val_loss, lr}``.

    Returns:
        Dicionário com ``categoria``, ``emoji``, ``titulo``, ``diagnostico``,
        ``sugestao``, ``checks`` (lista) e métricas resumidas.
    """
    if not loss_history:
        return {
            "categoria": "SEM_DADOS",
            "emoji": "⚪",
            "titulo": "SEM DADOS",
            "diagnostico": "Histórico de loss vazio.",
            "sugestao": "Treine o modelo antes de diagnosticar.",
            "checks": [],
            "epochs_trained": 0,
            "best_epoch": 0,
            "best_val_loss": float("nan"),
            "final_train_loss": float("nan"),
            "final_val_loss": float("nan"),
            "gap_pct": float("nan"),
        }

    val = np.array([h["val_loss"] for h in loss_history], dtype=float)
    train = np.array([h["train_loss"] for h in loss_history], dtype=float)
    n = len(val)

    best_idx = int(np.argmin(val))
    best_epoch = loss_history[best_idx]["epoch"]
    best_val = float(val[best_idx])
    final_train = float(train[-1])
    final_val = float(val[-1])
    gap_pct = (final_val - final_train) / final_train * 100 if final_train > 0 else 0.0

    # --- Sinais auxiliares ---
    initial_val = float(val[0])
    limiar_alto = float(np.percentile(val, 80))  # percentil 80 do val_loss
    # Queda relativa até a época 20 (ou final, se treinou menos)
    idx20 = min(19, n - 1)
    queda_20 = (initial_val - val[idx20]) / initial_val if initial_val > 0 else 0.0

    # Val subindo enquanto train cai (final do treino vs melhor época)
    val_subindo = final_val > best_val * 1.05 and final_train <= train[best_idx]

    # Instabilidade nas últimas 10 épocas
    janela = val[-10:] if n >= 10 else val
    media_janela = float(np.mean(janela))
    std_janela = float(np.std(janela))
    instavel = media_janela > 0 and std_janela > 0.2 * media_janela

    def base(epochs_trained: int) -> dict:
        return {
            "epochs_trained": n,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "final_train_loss": final_train,
            "final_val_loss": final_val,
            "gap_pct": gap_pct,
        }

    # 1) UNDERFITTING
    if final_val > limiar_alto or (n >= 20 and queda_20 < 0.10):
        return {
            "categoria": "UNDERFITTING",
            "emoji": "🟠",
            "titulo": "UNDERFITTING",
            "diagnostico": "Loss ainda alto — modelo não aprendeu suficiente",
            "sugestao": "Tente aumentar HIDDEN_SIZE, NUM_LAYERS ou EPOCHS. "
            "Verifique se as features estão corretas.",
            "checks": [
                "⚠️ Loss de validação permanece elevado",
                "⚠️ Queda insuficiente do loss inicial",
            ],
            **base(n),
        }

    # 2) OVERFITTING
    if gap_pct / 100.0 > 0.5 and val_subindo:
        return {
            "categoria": "OVERFITTING",
            "emoji": "🔴",
            "titulo": "OVERFITTING",
            "diagnostico": "Overfitting detectado — modelo memorizou o treino",
            "sugestao": "Aumente DROPOUT, reduza HIDDEN_SIZE ou adicione mais dados.",
            "checks": [
                "⚠️ Gap treino/val acima de 50%",
                "⚠️ Val loss subindo enquanto train loss cai",
            ],
            **base(n),
        }

    # 3) INSTÁVEL
    if instavel:
        return {
            "categoria": "INSTAVEL",
            "emoji": "🟡",
            "titulo": "INSTÁVEL",
            "diagnostico": "Loss instável — oscilações excessivas",
            "sugestao": "Reduza LEARNING_RATE ou BATCH_SIZE. "
            "Verifique se há outliers nos dados.",
            "checks": [
                "⚠️ Alto desvio padrão do val_loss nas últimas épocas",
            ],
            **base(n),
        }

    # 4) BOAS CONDIÇÕES MAS PODE MELHORAR (early stop precoce)
    if best_epoch < 30:
        return {
            "categoria": "EARLY_STOP_PRECOCE",
            "emoji": "🟢",
            "titulo": "BOM, MAS PODE MELHORAR",
            "diagnostico": "Treino encerrado cedo — pode haver espaço para melhorar",
            "sugestao": "Tente aumentar PATIENCE ou reduzir LEARNING_RATE para "
            "convergência mais lenta.",
            "checks": [
                "✅ Loss de treino e val decaem juntas",
                "ℹ️ Melhor época antes da 30 — convergência rápida",
            ],
            **base(n),
        }

    # 5) BOM APRENDIZADO (default)
    return {
        "categoria": "BOM_APRENDIZADO",
        "emoji": "🟢",
        "titulo": "BOM APRENDIZADO",
        "diagnostico": "Aprendizado saudável detectado",
        "sugestao": "Modelo pronto. Avalie métricas no test set antes de usar em "
        "produção.",
        "checks": [
            "✅ Loss de treino e val decaem juntas",
            "✅ Gap treino/val estável e pequeno",
            "✅ Sem sinais de overfitting",
        ],
        **base(n),
    }
