# MA Forecast — Pipeline Completo de Previsão de Média Móvel com PyTorch

## Visão Geral

Pipeline end-to-end que, dado um ativo, período da MA, timeframe e horizonte de previsão, executa automaticamente todas as etapas: coleta de dados do MT5, feature engineering, treinamento da rede neural LSTM Seq2Seq, diagnóstico de aprendizado e geração de gráficos. O usuário configura os parâmetros e roda um único comando.

**Comando único de exemplo:**
```bash
python main.py --symbol EURUSD --timeframe H1 --ma-period 20 --forecast-steps 5
```

---

## Stack Tecnológica

- **Python 3.10+**
- **PyTorch 2.x** — modelo LSTM Seq2Seq
- **MetaTrader5** — coleta de dados OHLCV direto do terminal MT5
- **pandas / numpy** — feature engineering
- **scikit-learn** — normalização (MinMaxScaler sem data leakage)
- **matplotlib** — todos os gráficos de diagnóstico
- **joblib** — serialização do scaler

---

## Estrutura de Arquivos

```
ma_forecast/
├── CLAUDE.md
├── config.py                  # parâmetros padrão (sobrescríveis via CLI)
├── data/
│   ├── collector.py           # coleta OHLCV do MT5
│   ├── features.py            # cálculo de todas as features
│   └── dataset.py             # PyTorch Dataset + sliding window + split
├── model/
│   ├── lstm_seq2seq.py        # arquitetura LSTM encoder-decoder
│   └── train.py               # loop de treino com early stopping
├── evaluate/
│   └── metrics.py             # MAE, RMSE, MAPE, acurácia direcional
├── diagnostics/
│   └── plots.py               # TODOS os gráficos do pipeline
├── predict/
│   ├── inference.py           # inferência com modelo treinado
│   └── mt5_simulation.py      # simula inferência ONNX como o EA MQL5 fará
├── export/
│   └── onnx_export.py         # exporta modelo treinado para ONNX
├── pipeline/
│   └── runner.py              # orquestrador do pipeline completo
├── artifacts/                 # modelos, scalers, datasets, gráficos (gitignore)
│   └── .gitkeep
├── requirements.txt
└── main.py                    # entry point CLI
```

---

## config.py — Parâmetros Padrão

```python
# Ativo e coleta
SYMBOL          = "EURUSD"
TIMEFRAME       = "H1"        # M5, M15, M30, H1, H4, D1
BARS_HISTORY    = 5000        # barras históricas a coletar

# Média móvel alvo
MA_PERIOD       = 20          # 20 ou 50

# Previsão
FORECAST_STEPS  = 5           # barras à frente (5 a 10)

# Janela de entrada do modelo
LOOKBACK_WINDOW = 60          # barras de contexto para o encoder

# Arquitetura
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.2

# Treino
EPOCHS          = 100
BATCH_SIZE      = 64
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-5
PATIENCE        = 15          # early stopping
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
# test = restante

# Backtest overlay
BACKTEST_BARS   = 200             # barras do test set usadas no plot 08

# Paths (gerados dinamicamente pelo runner com base nos parâmetros)
ARTIFACT_DIR    = "artifacts"
```

---

## pipeline/runner.py — Orquestrador Central

Este é o coração do sistema. Executa as etapas em sequência, com logging claro de cada fase, e para com mensagem de erro descritiva se qualquer etapa falhar.

### Etapas em ordem

```
[ETAPA 1/9] Conectando ao MT5 e coletando dados...
[ETAPA 2/9] Calculando features e salvando dataset...
[ETAPA 3/9] Criando janelas deslizantes e splits...
[ETAPA 4/9] Treinando modelo...
[ETAPA 5/9] Avaliando no conjunto de teste...
[ETAPA 6/9] Exportando modelo para ONNX...
[ETAPA 7/9] Gerando gráficos de diagnóstico...
[ETAPA 8/9] Gerando gráfico de comparação com preço real...
[ETAPA 9/9] Simulando inferência MT5 com ONNX Runtime...
✅ Pipeline concluído. Artefatos salvos em: artifacts/EURUSD_H1_MA20_F5/
```

### Geração de paths dinâmicos

Todos os artefatos são salvos em um subdiretório com nome baseado nos parâmetros:
```python
run_dir = f"artifacts/{symbol}_{timeframe}_MA{ma_period}_F{forecast_steps}/"
# Exemplo: artifacts/EURUSD_H1_MA20_F5/
```

Dentro de `run_dir`:
```
dataset_raw.csv          # OHLCV bruto coletado do MT5
dataset_features.csv     # dataset com todas as features calculadas
model.pt                 # state_dict do melhor modelo (PyTorch)
model.onnx               # modelo exportado para uso no MQL5
scaler.pkl               # MinMaxScaler serializado
loss_history.json        # loss por época (train e val)
metrics_test.json        # métricas finais no test set
training_config.json     # hiperparâmetros usados no treino (lr, batch, dropout)
plots/
  ├── 01_raw_price.png              # close e MA histórica
  ├── 02_features.png               # painel com todas as features
  ├── 03_dataset_split.png          # visualização do split treino/val/test
  ├── 04_training_loss.png          # curvas de loss com diagnóstico
  ├── 05_predictions_test.png       # MA real vs prevista no test set
  ├── 06_error_distribution.png     # histograma dos erros
  ├── 07_directional_accuracy.png   # acurácia direcional por horizonte
  ├── 08_backtest_overlay.png       # previsão da NN sobreposta ao gráfico real
  └── 09_mt5_simulation.png         # simulação de inferência ONNX como no MT5
```

---

## data/collector.py — Coleta do MT5

### Responsabilidades
- Verificar se o terminal MT5 está aberto; se não estiver, lançar erro claro
- Verificar se o símbolo existe no MT5; se não, listar símbolos disponíveis similares
- Baixar `BARS_HISTORY` barras do ativo e timeframe configurados
- Remover barras com volume zero (finais de semana, feriados)
- Retornar DataFrame com colunas: `time, open, high, low, close, tick_volume`
- Fechar conexão MT5 com `mt5.shutdown()` no bloco `finally`
- Salvar resultado em `run_dir/dataset_raw.csv`

### Mapeamento de timeframes
```python
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}
```

### Validações obrigatórias
- Mínimo de `LOOKBACK_WINDOW + MA_PERIOD + FORECAST_STEPS + 100` barras válidas após limpeza
- Se insuficiente: erro com sugestão de aumentar `BARS_HISTORY` ou usar timeframe menor

---

## data/features.py — Feature Engineering

### Features calculadas sobre o DataFrame cru

Todas as normalizações por ATR garantem comparabilidade entre ativos e períodos de volatilidade diferentes. Nenhuma feature usa a lib `ta` — tudo implementado com numpy/pandas puro para evitar dependências problemáticas no Windows.

| # | Feature | Cálculo | Coluna |
|---|---|---|---|
| 1 | MA alvo | `close.rolling(MA_PERIOD).mean()` | `ma` |
| 2 | ATR 14 | rolling max(H-L, \|H-Cp\|, \|L-Cp\|) janela 14 | `atr` |
| 3 | Retorno log | `log(close / close.shift(1))` | `log_return` |
| 4 | ATR normalizado | `atr / close` | `atr_norm` |
| 5 | RSI 14 | Wilder RSI(14) / 100 | `rsi` |
| 6 | Distância close→MA | `(close - ma) / atr` | `dist_ma` |
| 7 | Inclinação da MA | `(ma - ma.shift(1)) / atr` | `slope_ma` |
| 8 | Range do candle | `(high - low) / atr` | `hl` |
| 9 | Corpo do candle | `(close - open) / atr` | `body` |
| 10 | Seno da hora UTC | `sin(2π * hour / 24)` | `sin_hour` |
| 11 | Cosseno da hora UTC | `cos(2π * hour / 24)` | `cos_hour` |

**Total:** 11 features (9 em D1, sem `sin_hour` e `cos_hour`).

### Target (coluna `y`)

Target fixo no modo **delta normalizado por ATR:**

```
y = (ma.shift(-FORECAST_STEPS) - ma) / atr
```

Estacionário e adimensional — escala bem entre ativos e períodos de volatilidade distintos. O sinal já indica direção (positivo = MA sobe, negativo = MA desce). Não requer scaler separado.

Na inferência, converter de volta para preço: `ma_prevista = ma_atual + (y_pred * atr_atual)`

**Nota:** as colunas `ma` e `atr` brutas **não entram** como features de entrada do modelo (X) — têm unidade de preço e não estão normalizadas. Usar apenas as derivadas normalizadas (features 3 a 11).

- Remover linhas com NaN após todos os cálculos (primeiras `MA_PERIOD + 14` linhas, mais `FORECAST_STEPS` linhas do final por causa do `shift(-H)`)
- Se `TIMEFRAME == "D1"`: não calcular `sin_hour` e `cos_hour`
- Salvar DataFrame completo (features + target) em `run_dir/dataset_features.csv`
- Logar: `"Features calculadas: N linhas, 11 colunas, target: delta/ATR, período: YYYY-MM-DD a YYYY-MM-DD"`

### Função pública
```python
def build_features(df_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calcula todas as features sobre o DataFrame OHLCV bruto.
    Retorna DataFrame com features prontas para o Dataset PyTorch.
    """
```

---

## data/dataset.py — Dataset e Splits

### Normalização — CRÍTICO: sem data leakage

**Regra:** o `MinMaxScaler` é fitado **apenas nas linhas do split de treino** e depois aplicado via `transform` em val e test.

```python
# CORRETO:
scaler.fit(train_data[feature_cols])
train_norm = scaler.transform(train_data[feature_cols])
val_norm   = scaler.transform(val_data[feature_cols])
test_norm  = scaler.transform(test_data[feature_cols])

# ERRADO — não fazer:
scaler.fit(full_data[feature_cols])  # vaza info de val/test para o treino
```

O target `y` (delta/ATR) é adimensional — não precisa de scaler separado.  
Salvar scaler das features em `run_dir/scaler.pkl`.

### Sliding window

Para cada índice `i` válido no array normalizado:
- **X:** features das barras `[i : i + LOOKBACK_WINDOW]` → shape `(LOOKBACK_WINDOW, num_features)`
- **y:** delta normalizado calculado na barra `i + LOOKBACK_WINDOW` → shape `(1,)` — escalar por janela

### Split temporal (nunca embaralhar série temporal)
```
Treino → primeiros 70% das janelas
Val    → próximos 15%
Test   → últimos 15%
```

DataLoaders:
- `train_loader`: `shuffle=True`, `batch_size=BATCH_SIZE`
- `val_loader`: `shuffle=False`, `batch_size=BATCH_SIZE`
- `test_loader`: `shuffle=False`, `batch_size=BATCH_SIZE`

---

## model/lstm_seq2seq.py — Arquitetura

### Encoder
- `nn.LSTM(input_size=num_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, batch_first=True)`
- Entrada: `(batch, LOOKBACK_WINDOW, num_features)`
- Saída: `(hidden, cell)` para inicializar o decoder

### Decoder
- `nn.LSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, batch_first=True)`
- `nn.Linear(HIDDEN_SIZE, 1)` — projeta hidden → valor MA
- Geração autoregressiva passo a passo (1 token por vez)

### Teacher Forcing
```python
# Durante treino (target disponível):
use_teacher = random.random() < teacher_forcing_ratio  # ratio = 0.5
next_input = target[:, t] if use_teacher else prediction_t

# Durante inferência (target = None):
next_input = prediction_t  # sempre usa própria previsão
```

### Forward
```python
def forward(
    self,
    x: torch.Tensor,           # (batch, LOOKBACK_WINDOW, num_features)
    target: torch.Tensor = None,  # (batch, FORECAST_STEPS) — None na inferência
    teacher_forcing_ratio: float = 0.5
) -> torch.Tensor:              # (batch, FORECAST_STEPS)
```

---

## model/train.py — Loop de Treino

### Setup
- Detectar device: `cuda` > `mps` (Apple Silicon) > `cpu`
- Seed fixo: `torch.manual_seed(42)`, `np.random.seed(42)`
- Loss: `nn.MSELoss()`
- Optimizer: `Adam(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)`
- Scheduler: `ReduceLROnPlateau(mode='min', factor=0.5, patience=7)`
- Grad clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)`

### Early Stopping
- Monitorar `val_loss`
- Salvar `best_model_state` quando `val_loss` melhora
- Parar após `PATIENCE` épocas sem melhora
- Restaurar `best_model_state` ao final

### Log por época
```
[Época  12/100] train_loss=0.000312 | val_loss=0.000401 | lr=0.001000 | ⏱ 2.3s
```

Marcar com `🔥 melhor modelo` quando val_loss bater novo mínimo.

### Salvar ao final
- `run_dir/model.pt` — state_dict do melhor modelo
- `run_dir/loss_history.json` — lista de dicts `{epoch, train_loss, val_loss, lr}`

---

## evaluate/metrics.py

Calcular sobre valores **desnormalizados** (preços reais da SMA):

```python
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true e y_pred: shape (N, FORECAST_STEPS) — valores reais desnormalizados.
    Retorna dict com métricas por horizonte e métricas globais.
    """
```

### Métricas por horizonte (h=1 até FORECAST_STEPS)
- MAE em pips (para EURUSD, 1 pip = 0.0001)
- RMSE em pips
- Acurácia direcional — % de vezes que acertou subida/descida vs valor atual da MA

### Métricas globais (média sobre todos os horizontes)
- MAE global, RMSE global, Direcional global

### Saída salva em `run_dir/metrics_test.json`:
```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "ma_period": 20,
  "forecast_steps": 5,
  "by_horizon": [
    {"h": 1, "mae_pips": 4.2, "rmse_pips": 5.8, "directional_acc": 0.61},
    ...
  ],
  "global": {
    "mae_pips": 6.1,
    "rmse_pips": 8.3,
    "directional_acc": 0.58
  }
}
```

---

## diagnostics/plots.py — Todos os Gráficos

Cada função salva um PNG em `run_dir/plots/`. Usar `matplotlib` com estilo `seaborn-v0_8-darkgrid`. Título de cada gráfico inclui o run ID: `EURUSD | H1 | SMA20 | Forecast +5`.

---

### Plot 01 — Preço e MA Histórica (`01_raw_price.png`)

**O que mostra:** contexto dos dados coletados.
- Subplot superior: série de `close` (cinza claro) + `ma_target` (azul) ao longo do tempo
- Subplot inferior: volume (tick_volume) em barras verticais
- Marcar com linhas verticais pontilhadas onde começam val e test
- Legenda: datas de início de cada split

---

### Plot 02 — Painel de Features (`02_features.png`)

**O que mostra:** qualidade e comportamento das features calculadas.
- Grid de subplots (um por feature ativa): `log_return`, `atr_norm`, `rsi`, `dist_ma`, `slope_ma`, `hl`, `body`, `sin_hour`, `cos_hour`
- Cada subplot: série temporal + linha horizontal em zero ou em 0.5 quando relevante
- Título de cada subplot inclui média e desvio padrão da feature
- Subplot extra para o target `y`: distribuição (histograma) com linha vertical em zero — verificar se está centrado (sem viés sistemático)
- Detectar e anotar se alguma feature tem mais de 1% de NaN restante (warning visual em vermelho)

---

### Plot 03 — Visualização do Split (`03_dataset_split.png`)

**O que mostra:** como os dados foram divididos e distribuição dos targets.
- Subplot esquerdo: barra horizontal mostrando proporção treino/val/test com cores distintas + n° de janelas em cada split
- Subplot direito: distribuição (histograma + KDE) dos valores da `ma_target` em cada split — verificar se as três distribuições são similares (se não forem, avisar no título)

---

### Plot 04 — Curvas de Loss (`04_training_loss.png`)

**O que mostra:** se o modelo aprendeu (diagnóstico principal).

Layout: gráfico principal + painel de diagnóstico textual.

**Gráfico principal:**
- Loss de treino (azul) e validação (laranja) por época, escala log no eixo Y
- Linha vertical pontilhada verde na época do melhor modelo
- Sombra cinza na região após o early stopping (se aplicável)
- Anotação do valor de val_loss mínimo atingido

**Painel de diagnóstico** (texto estruturado ao lado ou abaixo):

Comparar os padrões clássicos de aprendizado e exibir o diagnóstico correspondente:

```
📊 DIAGNÓSTICO DE APRENDIZADO
─────────────────────────────────────────────
Épocas treinadas   : 47 / 100 (early stop na 47)
Melhor val_loss    : 0.000312 (época 32)
Loss final treino  : 0.000198
Loss final val     : 0.000401
Gap treino→val     : +27.9%

🔎 Padrão detectado: BOM APRENDIZADO
   ✅ Loss de treino e val decaem juntas
   ✅ Gap treino/val estável e pequeno
   ✅ Sem sinais de overfitting

💡 Sugestão: modelo pronto para produção.
─────────────────────────────────────────────
```

**Lógica de diagnóstico automático** (implementar em `diagnostics/learning_check.py`):

```python
def diagnose_learning(loss_history: list[dict]) -> dict:
    """
    Analisa o histórico de loss e retorna diagnóstico categorizado.
    """
```

Regras de classificação (verificar nesta ordem):

1. **UNDERFITTING**
   - Condição: `final_val_loss > limiar_alto` (definir como percentil 80 do val_loss inicial)
   - OU: val_loss não caiu mais de 10% do valor inicial após 20 épocas
   - Diagnóstico: `"Loss ainda alto — modelo não aprendeu suficiente"`
   - Sugestão: `"Tente aumentar HIDDEN_SIZE, NUM_LAYERS ou EPOCHS. Verifique se as features estão corretas."`

2. **OVERFITTING**
   - Condição: `(val_loss_final - train_loss_final) / train_loss_final > 0.5` (gap > 50%)
   - E: val_loss começou a subir enquanto train_loss continuou caindo
   - Diagnóstico: `"Overfitting detectado — modelo memorizou o treino"`
   - Sugestão: `"Aumente DROPOUT, reduza HIDDEN_SIZE ou adicione mais dados."`

3. **INSTÁVEL**
   - Condição: desvio padrão do val_loss nas últimas 10 épocas > 20% do val_loss médio nessas épocas
   - Diagnóstico: `"Loss instável — oscilações excessivas"`
   - Sugestão: `"Reduza LEARNING_RATE ou BATCH_SIZE. Verifique se há outliers nos dados."`

4. **BOAS CONDIÇÕES MAS PODE MELHORAR**
   - Condição: early stopping ativou antes da época 30
   - Diagnóstico: `"Treino encerrado cedo — pode haver espaço para melhorar"`
   - Sugestão: `"Tente aumentar PATIENCE ou reduzir LEARNING_RATE para convergência mais lenta."`

5. **BOM APRENDIZADO** (default se nenhuma condição acima for verdadeira)
   - Diagnóstico: `"Aprendizado saudável detectado"`
   - Sugestão: `"Modelo pronto. Avalie métricas no test set antes de usar em produção."`

---

### Plot 05 — MA Real vs Prevista no Test Set (`05_predictions_test.png`)

**O que mostra:** qualidade visual das previsões.
- Eixo X: tempo (índice das barras de teste)
- Linha azul: `ma_target` real
- Linha laranja pontilhada: previsão do modelo para `h=1` (primeiro passo)
- Banda sombreada laranja: intervalo entre `h=1` e `h=FORECAST_STEPS` (envelope das previsões)
- Subplot inferior: erro absoluto por barra (|real - previsto h=1|)
- Título inclui MAE e acurácia direcional globais

---

### Plot 06 — Distribuição dos Erros (`06_error_distribution.png`)

**O que mostra:** se os erros têm viés ou caudas pesadas.
- Histograma dos erros `(y_pred_h1 - y_true_h1)` desnormalizados em pips
- Linha KDE suavizada sobreposta
- Linha vertical em zero
- Anotação: média dos erros, desvio padrão, % de erros positivos vs negativos
- Se `|média dos erros| > 2 pips`: adicionar aviso `"⚠️ Viés detectado — modelo tende a superestimar/subestimar"`

---

### Plot 07 — Acurácia Direcional por Horizonte (`07_directional_accuracy.png`)

**O que mostra:** em quais horizontes o modelo é mais útil para trading.
- Gráfico de barras horizontais: um bar por horizonte (h=1 a h=FORECAST_STEPS)
- Valor em cada barra: acurácia direcional (%)
- Linha vertical vermelha em 50% (baseline aleatório)
- Colorir barras: verde se > 55%, amarelo se 50-55%, vermelho se < 50%
- Título: `"Acurácia Direcional — % de acerto na direção da MA"`

---

## export/onnx_export.py — Exportação para ONNX

### Por que ONNX para MQL5

O MetaTrader 5 suporta inferência de modelos ONNX nativamente desde o build 3683, via `OnnxCreate` / `OnnxRun`. Exportar o modelo treinado para ONNX permite rodá-lo diretamente no EA em MQL5, sem precisar de um servidor Python externo.

### Função principal

```python
def export_to_onnx(model: nn.Module, config: dict, run_dir: str) -> str:
    """
    Exporta o modelo LSTM treinado para formato ONNX.
    Retorna o path do arquivo .onnx gerado.
    """
```

### Procedimento de exportação

```python
model.eval()

# Input dummy com as dimensões exatas que o modelo espera
# batch_size=1 para inferência unitária no MQL5
dummy_input = torch.zeros(1, LOOKBACK_WINDOW, num_features)

onnx_path = os.path.join(run_dir, "model.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=12,          # compatível com MT5 build 3683+
    do_constant_folding=True,  # otimiza constantes em tempo de exportação
    input_names=["features"],
    output_names=["ma_delta"],
    dynamic_axes={
        "features":  {0: "batch_size"},
        "ma_delta":  {0: "batch_size"},
    }
)
```

### Validação pós-exportação

Após exportar, validar que o ONNX produz o mesmo resultado que o PyTorch para o mesmo input:

```python
import onnxruntime as ort

session = ort.InferenceSession(onnx_path)
ort_out = session.run(["ma_delta"], {"features": dummy_input.numpy()})[0]
pt_out  = model(dummy_input).detach().numpy()

max_diff = np.abs(ort_out - pt_out).max()
assert max_diff < 1e-5, f"Divergência ONNX vs PyTorch: {max_diff}"
logging.info(f"✅ ONNX validado — divergência máxima: {max_diff:.2e}")
```

Se a validação falhar, logar erro detalhado e não sobrescrever arquivo anterior.

### Salvar metadados para uso no MQL5

Salvar `run_dir/onnx_metadata.json` com as informações necessárias para replicar o pré-processamento no EA:

```json
{
  "symbol":          "EURUSD",
  "timeframe":       "H1",
  "ma_period":       20,
  "forecast_steps":  5,
  "lookback_window": 60,
  "num_features":    11,
  "feature_order":   ["log_return", "atr_norm", "rsi", "dist_ma", "slope_ma", "hl", "body", "sin_hour", "cos_hour"],
  "target":          "delta_atr",
  "scaler_min":      [...],
  "scaler_max":      [...],
  "opset_version":   12,
  "torch_version":   "2.1.0",
  "export_date":     "2025-06-01T14:00:00"
}
```

Os campos `scaler_min` e `scaler_max` são os arrays do `MinMaxScaler` fitado, exportados como listas — permitem replicar a normalização exatamente em MQL5 sem depender do arquivo `.pkl`.

### Log esperado

```
[ETAPA 6/8] Exportando modelo para ONNX...
  📐 Input shape : (1, 60, 11)
  📐 Output shape: (1, 1)
  ✅ Exportado  : artifacts/EURUSD_H1_MA20_F5/model.onnx
  ✅ Validado   : divergência máxima PyTorch↔ONNX = 3.21e-07
  💾 Metadados  : artifacts/EURUSD_H1_MA20_F5/onnx_metadata.json
```

---

### Plot 08 — Backtest Visual: NN vs Gráfico Real (`08_backtest_overlay.png`)

**O que mostra:** como a previsão da NN se comportaria na prática — comparação visual direta entre a MA calculada pelo MT5 e a MA prevista pelo modelo ao longo de um período contínuo do conjunto de teste.

**Período plotado:** últimas `BACKTEST_BARS = 200` barras do conjunto de teste (configurável). Representa aproximadamente as condições mais recentes de mercado vistas pelo modelo.

**Layout:** figura com 3 subplots verticais.

**Subplot 1 — Gráfico de preço com ambas as MAs (principal):**
- Candlesticks ou linha de close (cinza claro)
- MA real calculada (azul sólido, linha mais espessa)
- MA prevista pela NN (laranja pontilhado, linha mais fina)
- Área sombreada entre as duas curvas: verde quando NN > real (modelo otimista), vermelho quando NN < real (modelo pessimista)
- Legenda com símbolo, timeframe e período das barras

**Subplot 2 — Erro absoluto em pips:**
- Barra vertical por barra: `|ma_real - ma_prevista|` em pips
- Linha horizontal pontilhada no MAE global (referência)
- Colorir barras: verde se abaixo do MAE, vermelho se acima

**Subplot 3 — Acurácia direcional acumulada:**
- Linha mostrando a acurácia direcional cumulativa ao longo das barras plotadas
- Linha horizontal em 50% (baseline aleatório)
- Linha horizontal em 55% (threshold de relevância para trading)
- Anotação do valor final no canto direito

**Título do gráfico:**
```
EURUSD H1 | SMA 20 | NN Forecast +5 barras
Período: 2024-09-05 a 2025-06-01 | MAE: 6.1 pips | Dir. Acc: 58.2%
```

**Nota de implementação:** as previsões para este plot são geradas passando cada janela de teste pelo modelo no modo inferência (sem teacher forcing), reconvertendo o delta previsto para preço com `ma_prevista = ma_real[t] + (y_pred * atr[t])`, e alinhando temporalmente com as barras reais usando o índice `time` do DataFrame.

---



## predict/mt5_simulation.py — Simulação de Inferência como no MT5

Esta etapa replica **exatamente** o fluxo que o EA em MQL5 executará em produção: coleta as últimas barras disponíveis do MT5, monta o vetor de features da mesma forma que o EA fará, normaliza com os parâmetros do `onnx_metadata.json` (sem usar o `.pkl`), e roda o `model.onnx` via ONNX Runtime. É a validação final de ponta a ponta antes de portar o modelo para o MT5.

### Por que esta etapa é crítica

O backtesting do Plot 08 usa o scaler `.pkl` do scikit-learn e dados do conjunto de teste (histórico). Esta etapa usa o `onnx_metadata.json` e dados **ao vivo recém-coletados do MT5** — simulando exatamente o que o EA fará a cada nova barra. Se os resultados divergirem, há um bug na normalização ou na ordem das features.

### Função principal

```python
def run_mt5_simulation(run_dir: str, n_recent_bars: int = 300) -> dict:
    """
    Simula a inferência ONNX exatamente como o EA MQL5 fará em produção.
    Coleta barras recentes do MT5, aplica normalização via onnx_metadata.json,
    roda o modelo ONNX e retorna previsões + gráfico comparativo.
    """
```

### Fluxo interno passo a passo

**Passo 1 — Carregar metadados (sem usar .pkl)**
```python
with open(f"{run_dir}/onnx_metadata.json") as f:
    meta = json.load(f)

feature_order  = meta["feature_order"]   # lista com a ordem exata das features
scaler_min     = np.array(meta["scaler_min"])
scaler_max     = np.array(meta["scaler_max"])
lookback       = meta["lookback_window"]
ma_period      = meta["ma_period"]
forecast_steps = meta["forecast_steps"]
```

**Passo 2 — Coletar barras recentes do MT5**

Coletar `n_recent_bars + ma_period + 14 + 1` barras fechadas para ter margem suficiente para calcular todas as features. Remover barra atual (em andamento). Remover barras com volume zero.

**Passo 3 — Calcular features com numpy/pandas puro**

Aplicar exatamente as mesmas fórmulas de `data/features.py`, na mesma ordem de `feature_order`. Esta é a replicação do pré-processamento que o EA fará em MQL5.

```python
# Montar array de features na ordem correta
X_raw = df[feature_order].values  # shape: (n_barras, num_features)

# Usar apenas as últimas (lookback) barras
X_window = X_raw[-lookback:]      # shape: (lookback, num_features)
```

**Passo 4 — Normalizar com scaler_min/scaler_max do metadata**

```python
# MinMax manual — replica o que o EA fará em MQL5
X_norm = (X_window - scaler_min) / (scaler_max - scaler_min + 1e-8)
X_input = X_norm.astype(np.float32).reshape(1, lookback, len(feature_order))
```

**Passo 5 — Rodar ONNX Runtime**

```python
session = ort.InferenceSession(f"{run_dir}/model.onnx")
y_pred_delta = session.run(["ma_delta"], {"features": X_input})[0][0][0]
```

**Passo 6 — Desnormalizar para preço**

```python
ma_atual  = df["ma"].iloc[-1]
atr_atual = df["atr"].iloc[-1]
ma_prevista = ma_atual + (y_pred_delta * atr_atual)
direction   = "UP" if y_pred_delta > 0 else "DOWN"
```

**Passo 7 — Retornar resultado**

```python
{
    "symbol":         meta["symbol"],
    "timeframe":      meta["timeframe"],
    "ma_period":      ma_period,
    "forecast_steps": forecast_steps,
    "last_bar_time":  str(df["time"].iloc[-1]),
    "current_close":  float(df["close"].iloc[-1]),
    "current_ma":     float(ma_atual),
    "current_atr":    float(atr_atual),
    "y_pred_delta":   float(y_pred_delta),
    "ma_forecast":    float(ma_prevista),
    "direction":      direction,
    "generated_at":   datetime.utcnow().isoformat()
}
```

### Plot 09 — Simulação MT5 (`09_mt5_simulation.png`)

**O que mostra:** o ponto exato onde o modelo está "olhando" e para onde ele está prevendo a MA.

**Layout:** figura única com foco nas últimas `n_recent_bars` barras.

- Linha de close (cinza claro)
- MA real das últimas barras (azul sólido)
- Marcador vertical pontilhado na última barra conhecida (linha verde `"agora"`)
- Ponto único à direita: `ma_forecast` para `+FORECAST_STEPS` barras (estrela laranja)
- Seta anotada entre `ma_atual` e `ma_forecast` mostrando o delta em pips e a direção
- Caixa de texto no canto com o resultado completo:
```
┌─────────────────────────────┐
│ ONNX Simulation — EURUSD H1 │
│ Barra atual : 2025-06-01 14h│
│ MA atual    : 1.08234       │
│ MA prevista : 1.08291 (+5b) │
│ Delta       : +5.7 pips ↑   │
│ Direção     : UP            │
│ ATR atual   : 0.00087       │
└─────────────────────────────┘
```

- Título: `"Simulação de Inferência ONNX — como o EA MQL5 verá este modelo"`

### Log esperado no terminal

```
[ETAPA 9/9] Simulando inferência MT5 com ONNX Runtime...
  📊 Barras coletadas   : 314 (últimas barras ao vivo)
  🕐 Última barra       : 2025-06-01 14:00:00
  📈 Close atual        : 1.08198
  〰  MA atual (SMA 20)  : 1.08234
  ⚡ ATR atual          : 0.00087
  🔮 y_pred (delta/ATR) : +0.0657
  🎯 MA prevista (+5b)  : 1.08291
  📐 Variação           : +5.7 pips ↑ UP
  ✅ 09_mt5_simulation.png
```

---

## predict/inference.py

```python
def predict_next(run_dir: str) -> dict:
    """
    Carrega modelo e scaler do run_dir, coleta dados recentes do MT5,
    e retorna previsão das próximas FORECAST_STEPS barras da MA.
    Usa ONNX Runtime para inferência (mesmo engine que o MT5).
    """
```

**Retorno:**
```python
{
    "symbol": "EURUSD",
    "timeframe": "H1",
    "ma_period": 20,
    "forecast_steps": 5,
    "current_ma": 1.08234,
    "forecast": [
        {"bar": 1, "ma_value": 1.08251},
        {"bar": 2, "ma_value": 1.08268},
        {"bar": 3, "ma_value": 1.08291},
        {"bar": 4, "ma_value": 1.08315},
        {"bar": 5, "ma_value": 1.08302},
    ],
    "direction": "UP",
    "generated_at": "2025-06-01 14:00:00 UTC"
}
```

---

## main.py — Entry Point CLI

### Argumentos

```bash
python main.py [--symbol STR] [--timeframe STR] [--ma-period INT]
               [--forecast-steps INT] [--bars INT] [--retrain]
               [--lr FLOAT] [--batch-size INT] [--dropout FLOAT]
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--symbol` | `EURUSD` | Ativo MT5 |
| `--timeframe` | `H1` | Timeframe |
| `--ma-period` | `20` | Período da SMA |
| `--forecast-steps` | `5` | Horizonte de previsão |
| `--bars` | `5000` | Barras históricas |
| `--retrain` | `False` | Força novo treino mesmo se modelo existir (flag, sem valor) |
| `--lr` | `0.001` | Learning rate (sobrescreve config) |
| `--batch-size` | `64` | Batch size (sobrescreve config) |
| `--dropout` | `0.2` | Dropout (sobrescreve config) |

### Lógica de detecção de modelo existente

Ao iniciar, o runner verifica se o `run_dir` correspondente já possui os três artefatos obrigatórios:

```python
model_exists = all([
    os.path.exists(f"{run_dir}/model.onnx"),
    os.path.exists(f"{run_dir}/onnx_metadata.json"),
    os.path.exists(f"{run_dir}/scaler.pkl"),
])
```

**Se modelo não existe:** executa o pipeline completo (etapas 1 a 9).

**Se modelo existe e `--retrain` não foi passado:** exibe resumo do modelo salvo e pergunta interativamente:

```
============================================================
  MA FORECAST PIPELINE
  EURUSD | H4 | SMA 20 | Forecast +5 barras
============================================================

⚠️  Modelo já treinado encontrado em: artifacts/EURUSD_H4_MA20_F5/
   Treinado em : 2025-05-30 11:42:18
   MAE test    : 3.02 pips
   Dir. Acc    : 90.3%
   ONNX        : model.onnx (287 KB)
   Diagnóstico : 🟡 INSTÁVEL (gap treino→val: +111.8%)

   Hiperparâmetros usados no treino anterior:
   LEARNING_RATE : 0.001000
   BATCH_SIZE    : 64
   DROPOUT       : 0.200

   O que deseja fazer?
   [1] Usar modelo existente → gerar apenas inferência e gráficos
   [2] Treinar novamente     → manter hiperparâmetros atuais
   [3] Treinar novamente     → ajustar hiperparâmetros antes
   [4] Cancelar

Escolha (1/2/3/4): _
```

**Se usuário escolhe 1:** pula etapas 1 a 8, executa apenas a etapa 9 (simulação MT5) e gera o `09_mt5_simulation.png` atualizado com dados ao vivo.

**Se usuário escolhe 2:** executa pipeline completo com os mesmos hiperparâmetros, sobrescrevendo artefatos anteriores.

**Se usuário escolhe 3:** exibe prompt de ajuste de hiperparâmetros antes de iniciar o treino:

```
   Ajuste de hiperparâmetros (Enter = manter valor atual):

   LEARNING_RATE [atual: 0.001000] → _
   BATCH_SIZE    [atual: 64      ] → _
   DROPOUT       [atual: 0.200   ] → _

   ──────────────────────────────────────
   Hiperparâmetros confirmados:
   LEARNING_RATE : 0.000500  ← alterado
   BATCH_SIZE    : 32        ← alterado
   DROPOUT       : 0.300     ← alterado

   Confirmar e iniciar treino? (s/n): _
```

Após confirmação, executa pipeline completo com os novos valores. Os hiperparâmetros usados são salvos em `run_dir/training_config.json` ao final do treino para referência futura.

**Se usuário escolhe 4:** encerra sem fazer nada.

**Se modelo existe e `--retrain` foi passado:** pula o prompt e vai direto para o pipeline completo com os hiperparâmetros do config (ou os passados via `--lr`, `--batch-size`, `--dropout`). Útil para automação.

```bash
# Retreinar sem prompt com hiperparâmetros ajustados
python main.py --symbol EURUSD --timeframe H4 --ma-period 20 --forecast-steps 5 \
               --retrain --lr 0.0005 --batch-size 32 --dropout 0.3
```

### Fluxo quando modelo existe e usuário escolhe opção 1

```
============================================================
  MA FORECAST PIPELINE
  EURUSD | H4 | SMA 20 | Forecast +5 barras
  Run dir: artifacts/EURUSD_H4_MA20_F5/
============================================================

✅ Modelo existente carregado (treinado em 2025-05-30 11:42:18)
⏭  Pulando etapas 1 a 8 — executando apenas inferência ao vivo

[ETAPA 9/9] Simulando inferência MT5 com ONNX Runtime...
  📊 Barras coletadas   : 314 (últimas barras ao vivo)
  🕐 Última barra       : 2025-06-22 14:00:00
  📈 Close atual        : 1.08198
  〰  MA atual (SMA 20)  : 1.08234
  ⚡ ATR atual          : 0.00087
  🔮 y_pred (delta/ATR) : +0.0657
  🎯 MA prevista (+5b)  : 1.08291
  📐 Variação           : +5.7 pips ↑ UP
  ✅ 09_mt5_simulation.png

============================================================
  ✅ INFERÊNCIA CONCLUÍDA
  Tempo total: 8s
  Artefatos: artifacts/EURUSD_H4_MA20_F5/plots/09_mt5_simulation.png
============================================================
```

### Fluxo pipeline completo (modelo novo ou `--retrain`)
  ✅ 4987 barras válidas coletadas (2021-03-15 a 2025-06-01)
  💾 Salvo: artifacts/EURUSD_H1_MA20_F5/dataset_raw.csv

[ETAPA 2/9] Calculando features...
  ✅ 11 features calculadas | target: delta | 4937 linhas válidas
  💾 Salvo: artifacts/EURUSD_H1_MA20_F5/dataset_features.csv

[ETAPA 3/9] Criando janelas e splits...
  ✅ Janelas: 4891 total
     Treino : 3423 janelas (2021-03-15 a 2024-01-10)
     Val    :  734 janelas (2024-01-10 a 2024-09-05)
     Test   :  734 janelas (2024-09-05 a 2025-06-01)

[ETAPA 4/9] Treinando modelo (device: cuda)...
  [Época   1/100] train=0.001823 | val=0.001941 | lr=0.001000
  [Época   5/100] train=0.000891 | val=0.000934 | lr=0.001000 🔥
  ...
  [Época  47/100] train=0.000198 | val=0.000401 | lr=0.000500
  ⏹  Early stopping na época 47 (melhor: época 32)
  💾 Salvo: artifacts/EURUSD_H1_MA20_F5/model.pt

[ETAPA 5/9] Avaliando no test set...
  MAE global     : 6.1 pips
  RMSE global    : 8.3 pips
  Dir. Accuracy  : 58.2%
  💾 Salvo: artifacts/EURUSD_H1_MA20_F5/metrics_test.json

[ETAPA 6/9] Exportando modelo para ONNX...
  📐 Input shape : (1, 60, 11)
  📐 Output shape: (1, 1)
  ✅ Exportado  : artifacts/EURUSD_H1_MA20_F5/model.onnx
  ✅ Validado   : divergência máxima PyTorch↔ONNX = 3.21e-07
  💾 Metadados  : artifacts/EURUSD_H1_MA20_F5/onnx_metadata.json

[ETAPA 7/9] Gerando gráficos de diagnóstico...
  ✅ 01_raw_price.png
  ✅ 02_features.png
  ✅ 03_dataset_split.png
  ✅ 04_training_loss.png
  ✅ 05_predictions_test.png
  ✅ 06_error_distribution.png
  ✅ 07_directional_accuracy.png

[ETAPA 8/9] Gerando backtest overlay...
  ✅ 08_backtest_overlay.png (últimas 200 barras do test set)

[ETAPA 9/9] Simulando inferência MT5 com ONNX Runtime...
  📊 Barras coletadas   : 314 (últimas barras ao vivo)
  🕐 Última barra       : 2025-06-01 14:00:00
  📈 Close atual        : 1.08198
  〰  MA atual (SMA 20)  : 1.08234
  ⚡ ATR atual          : 0.00087
  🔮 y_pred (delta/ATR) : +0.0657
  🎯 MA prevista (+5b)  : 1.08291
  📐 Variação           : +5.7 pips ↑ UP
  ✅ 09_mt5_simulation.png

============================================================
  ✅ PIPELINE CONCLUÍDO
  Tempo total: 4m 41s
  Artefatos: artifacts/EURUSD_H1_MA20_F5/
  Diagnóstico: 🟢 BOM APRENDIZADO
============================================================
```

---

## diagnostics/learning_check.py

Módulo separado com a função `diagnose_learning` descrita na seção do Plot 04. Importado tanto pelo `plots.py` quanto pelo `runner.py` para exibir o diagnóstico no terminal ao final do pipeline.

---

## requirements.txt

```
torch>=2.1.0
MetaTrader5>=5.0.45
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
onnx>=1.15.0
onnxruntime>=1.17.0
onnxscript>=0.1.0
```

---

## Convenções de Código

- **Idioma:** código em inglês, comentários e docstrings em **português brasileiro**
- **Type hints** em todas as funções públicas
- **Docstrings** com descrição, parâmetros (`Args:`) e retorno (`Returns:`)
- Logging via `logging.getLogger(__name__)`, nível `INFO` por padrão
- Funções com responsabilidade única (máx. ~50 linhas)
- Nenhuma lógica de negócio em `main.py` — apenas parse de args e chamada ao `runner`

---

## Checklist de Qualidade

- [ ] Scaler fitado **somente** no split de treino
- [ ] Split é **temporal** — `DataLoader` de val/test com `shuffle=False`
- [ ] Teacher forcing desligado (`ratio=0.0`) durante avaliação e inferência
- [ ] Métricas calculadas sobre valores **desnormalizados**
- [ ] `mt5.shutdown()` chamado em bloco `finally`
- [ ] `model.eval()` + `torch.no_grad()` em toda avaliação e inferência
- [ ] Seeds fixos: `torch.manual_seed(42)`, `np.random.seed(42)`
- [ ] Barras de fim de semana (volume zero) removidas antes de qualquer cálculo
- [ ] Gráficos salvos com `dpi=150` e `bbox_inches='tight'`
- [ ] Run dir criado antes de qualquer tentativa de salvar arquivo
- [ ] ONNX validado contra PyTorch antes de salvar (`max_diff < 1e-5`)
- [ ] `onnx_metadata.json` inclui `scaler_min` e `scaler_max` como listas (para replicar normalização em MQL5)
- [ ] Simulação MT5 (Etapa 9) usa **somente** `onnx_metadata.json` para normalizar — nunca o `.pkl`
- [ ] Ordem das features na simulação bate exatamente com `feature_order` do metadata
- [ ] `training_config.json` salvo ao final de todo treino com `lr`, `batch_size`, `dropout` e `trained_at`
- [ ] Menu interativo lê `training_config.json` para exibir hiperparâmetros do treino anterior

---

## Notas de Implementação

- O terminal MT5 **deve estar aberto e com conta logada** antes de executar
- Para `TIMEFRAME = "D1"`: não calcular `sin_hour` e `cos_hour` (sem variação intraday)
- O target é **delta normalizado por ATR:** `y = (ma[t+H] - ma[t]) / atr[t]`. Para converter de volta a preço na inferência: `ma_prevista = ma_atual + (y_pred * atr_atual)`
- As colunas `ma` e `atr` brutas **não entram** como features de entrada do modelo — apenas as derivadas normalizadas (features 3 a 11)
- Barras incompletas (barra atual em andamento) **não devem** ser incluídas nos dados de treino — usar apenas barras fechadas