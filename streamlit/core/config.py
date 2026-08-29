"""Parâmetros e caminhos padrão do painel de gerenciamento de redes neurais."""

import os

# Diretórios locais (relativos à raiz do painel streamlit/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_cache")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
REGISTRY_PATH = os.path.join(RUNS_DIR, "registry.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Timeframes suportados no MT5
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

# Símbolos sugeridos (o usuário pode digitar qualquer símbolo válido no MT5)
SYMBOLS_SUGERIDOS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURJPY", "GBPJPY", "XAUUSD", "US30", "US500", "USTEC",
]

# Prefixos usados no Firebase Storage
FIREBASE_DATASETS_PREFIX = "datasets/"
FIREBASE_MODELS_PREFIX = "models/"

# Padrões de coleta de dados
DEFAULT_BARS_HISTORY = 5000

# Padrões de features / dataset supervisionado
DEFAULT_LOOKBACK_WINDOW = 60
DEFAULT_HORIZON = 5

# Padrões de arquitetura
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2

# Padrões de treino
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_PATIENCE = 15
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
# teste = restante

DEFAULT_SEED = 42

ARQUITETURAS = ["LSTM", "GRU", "MLP"]
TAREFAS = [
    "Regressão (retorno futuro)",
    "Classificação (direção do preço)",
    "Classificação (pullback vs. continuação de tendência)",
    "Classificação (regime de mercado: baixa/lateral/alta)",
]

# Explicação curta de cada tarefa, mostrada na UI ao selecioná-la.
TAREFA_DESCRICOES = {
    "Regressão (retorno futuro)": (
        "O modelo tenta prever **um número**: quanto o preço deve variar (retorno "
        "percentual) daqui a N barras, com base na janela de contexto informada. "
        "Ex.: prever que o EURUSD deve subir 0,3% nas próximas 5 barras."
    ),
    "Classificação (direção do preço)": (
        "O modelo tenta prever **se o preço vai subir ou cair** daqui a N barras "
        "(probabilidade de alta, de 0 a 1) — não prevê o tamanho do movimento, só a direção."
    ),
    "Classificação (pullback vs. continuação de tendência)": (
        "Em momentos de **retração contra a tendência vigente** (pullback), o modelo "
        "tenta prever se o preço vai **retomar a tendência** (romper o topo/fundo anterior "
        "a favor dela) ou **reverter/falhar** (romper a estrutura contrária antes)."
    ),
    "Classificação (regime de mercado: baixa/lateral/alta)": (
        "O modelo tenta prever **qual regime vai predominar** nas próximas N barras: "
        "tendência de **baixa**, mercado **lateral** (sem tendência clara) ou tendência "
        "de **alta** — útil para adaptar a estratégia ao contexto do mercado."
    ),
}

# Padrões da tarefa de pullback/continuação (core/labeling.py)
DEFAULT_EMA_FAST = 20
DEFAULT_EMA_SLOW = 50
DEFAULT_SWING_ORDER = 5
DEFAULT_PULLBACK_HORIZON = 20
DEFAULT_MIN_RETRACEMENT = 0.001  # 0.1%

# Padrões da tarefa de regime de mercado (core/labeling.py)
REGIME_CLASSES = ["Baixa", "Lateral", "Alta"]  # índices 0, 1, 2
DEFAULT_REGIME_HORIZON = 20
DEFAULT_REGIME_VOL_WINDOW = 20
DEFAULT_REGIME_K_LATERAL = 1.0
