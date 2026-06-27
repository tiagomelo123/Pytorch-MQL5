import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    predictions = model.predict(x_test)
    probabilities = positive_class_probabilities(model, x_test)

    metrics: dict[str, object] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "positive_precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
    }

    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
    else:
        metrics["roc_auc"] = None

    return metrics


def positive_class_probabilities(model, x_data: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x_data)
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def print_metrics(metrics: dict[str, object]) -> None:
    print("Metricas de classificacao")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    roc_auc = metrics["roc_auc"]
    print(f"ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC AUC: indisponivel")
    print(f"Precision classe positiva: {metrics['positive_precision']:.4f}")
    print("Matriz de confusao:")
    print(np.array(metrics["confusion_matrix"]))
    print("Relatorio:")
    print(metrics["classification_report"])
