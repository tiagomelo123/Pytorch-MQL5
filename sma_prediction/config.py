"""Parâmetros padrão do pipeline de previsão de média móvel.

Todos os valores podem ser sobrescritos via CLI (ver ``main.py``). O dicionário
final de configuração é montado em ``main.py``/``pipeline.runner`` combinando
estes padrões com os argumentos passados pelo usuário.
"""

# Ativo e coleta
SYMBOL = "EURUSD"
TIMEFRAME = "H1"  # M1, M5, M15, M30, H1, H4, D1
BARS_HISTORY = 5000  # barras históricas a coletar

# Média móvel alvo
MA_PERIOD = 20  # 20 ou 50

# Previsão
FORECAST_STEPS = 5  # barras à frente (5 a 10)

# Janela de entrada do modelo
LOOKBACK_WINDOW = 60  # barras de contexto para o encoder

# Arquitetura
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Treino
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15  # early stopping
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test = restante (1 - TRAIN_RATIO - VAL_RATIO)

# Teacher forcing (treino)
TEACHER_FORCING_RATIO = 0.5

# Backtest overlay
BACKTEST_BARS = 200  # barras do test set usadas no plot 08

# Paths
ARTIFACT_DIR = "artifacts"

# Estilo dos gráficos
PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_DPI = 150

# Seed global
SEED = 42


def build_config(**overrides) -> dict:
    """Monta o dicionário de configuração a partir dos padrões deste módulo.

    Args:
        **overrides: pares chave/valor que sobrescrevem os padrões. Valores
            ``None`` são ignorados (mantém-se o padrão).

    Returns:
        Dicionário com todos os parâmetros de configuração em minúsculas.
    """
    config = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "bars_history": BARS_HISTORY,
        "ma_period": MA_PERIOD,
        "forecast_steps": FORECAST_STEPS,
        "lookback_window": LOOKBACK_WINDOW,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "teacher_forcing_ratio": TEACHER_FORCING_RATIO,
        "backtest_bars": BACKTEST_BARS,
        "artifact_dir": ARTIFACT_DIR,
        "plot_style": PLOT_STYLE,
        "plot_dpi": PLOT_DPI,
        "seed": SEED,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config
