import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_metrics_report(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def save_learning_curve(model, output_path: Path) -> bool:
    mlp = get_mlp_estimator(model)
    loss_curve = getattr(mlp, "loss_curve_", None)
    validation_scores = getattr(mlp, "validation_scores_", None)

    if not loss_curve:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(loss_curve) + 1)

    fig, ax_loss = plt.subplots(figsize=(10, 5))
    ax_loss.plot(epochs, loss_curve, label="Train loss", color="#1f77b4", linewidth=2)
    ax_loss.set_title("Curva de aprendizagem - MLPClassifier")
    ax_loss.set_xlabel("Epoca")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.25)

    lines, labels = ax_loss.get_legend_handles_labels()
    if validation_scores is not None and len(validation_scores) == len(loss_curve):
        ax_score = ax_loss.twinx()
        ax_score.plot(
            epochs,
            validation_scores,
            label="Validation score",
            color="#2ca02c",
            linewidth=2,
        )
        ax_score.set_ylabel("Validation score")
        score_lines, score_labels = ax_score.get_legend_handles_labels()
        lines += score_lines
        labels += score_labels

    ax_loss.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def get_mlp_estimator(model):
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Tipo nao serializavel: {type(value)!r}")
