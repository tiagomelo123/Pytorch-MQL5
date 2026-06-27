import argparse

import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    SYMBOL,
    TIMEFRAME,
    learning_curve_path,
    metrics_report_path,
    model_path,
    processed_data_path,
    TRAIN_SIZE,
)
from .evaluate import evaluate_classifier, print_metrics
from .reporting import save_learning_curve, save_metrics_report


def load_processed_dataset(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset processado nao encontrado em {path}. Rode: python -m src.features"
        )
    return pd.read_csv(path, parse_dates=["time"])


def temporal_train_test_split(
    df: pd.DataFrame, train_size: float = TRAIN_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_size < 1:
        raise ValueError("train_size deve estar entre 0 e 1.")
    split_index = int(len(df) * train_size)
    if split_index == 0 or split_index == len(df):
        raise ValueError("Dataset pequeno demais para separar treino e teste.")
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def build_neural_network(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, object]]:
    usable = df[df["label"] != -1].copy()
    if usable.empty:
        raise ValueError("Nao ha registros com label 0/1 para treinar.")

    missing_features = [column for column in FEATURE_COLUMNS if column not in usable.columns]
    if missing_features:
        raise ValueError(f"Features ausentes no dataset: {missing_features}")

    train_df, test_df = temporal_train_test_split(usable)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].astype(int)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("O conjunto de treino precisa conter labels 0 e 1.")

    model = build_neural_network()
    model.fit(x_train, y_train)
    metrics = evaluate_classifier(model, x_test, y_test)
    metrics["train_rows"] = len(train_df)
    metrics["test_rows"] = len(test_df)
    metrics["train_label_counts"] = y_train.value_counts().sort_index().to_dict()
    metrics["test_label_counts"] = y_test.value_counts().sort_index().to_dict()
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina a rede neural TP/SL.")
    parser.add_argument("--symbol", default=SYMBOL, help="Ativo. Ex: EURUSD, GBPUSD, USDJPY.")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe. Ex: M1, M5, M15, H1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = processed_data_path(args.symbol, args.timeframe)
    output_model_path = model_path(args.symbol, args.timeframe)
    output_metrics_path = metrics_report_path(args.symbol, args.timeframe)
    output_curve_path = learning_curve_path(args.symbol, args.timeframe)

    df = load_processed_dataset(dataset_path)
    model, metrics = train_model(df)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model_path)
    save_metrics_report(metrics, output_metrics_path)
    curve_saved = save_learning_curve(model, output_curve_path)

    print_metrics(metrics)
    print(f"Modelo salvo em: {output_model_path}")
    print(f"Metricas salvas em: {output_metrics_path}")
    if curve_saved:
        print(f"Grafico de aprendizagem salvo em: {output_curve_path}")
    else:
        print("Grafico de aprendizagem indisponivel para este modelo.")


if __name__ == "__main__":
    main()
