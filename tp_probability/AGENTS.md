# AGENTS.md

## Objetivo do projeto

Criar um sistema Python para classificar se uma operação de Forex atinge o Take Profit antes do Stop Loss.

O modelo não deve tentar prever o preço exato do ativo. O objetivo é estimar a probabilidade de uma entrada atingir o TP antes do SL, considerando o contexto atual do mercado.

Exemplo de pergunta que o modelo deve responder:

> Dado o contexto atual do EUR/USD no timeframe M5, uma entrada comprada com TP de 20 pips e SL de 15 pips tem boa chance de bater o Take Profit antes do Stop Loss?

---

## Escopo inicial

Começar com um MVP simples e bem testável:

- Ativo: EUR/USD
- Timeframe inicial: M5
- Operações: BUY primeiro
- TP inicial: 20 pips
- SL inicial: 15 pips
- Horizonte máximo: 50 candles futuros
- Modelo inicial: `MLPClassifier`, `RandomForestClassifier` ou outro classificador simples do scikit-learn

Depois do MVP, permitir expansão para:

- SELL
- múltiplos pares
- múltiplos timeframes
- LightGBM
- XGBoost
- LSTM
- GRU
- 1D CNN
- Transformers temporais

---

## Stack

Usar preferencialmente:

- Python 3.11+
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib

Bibliotecas opcionais para fases futuras:

- lightgbm
- xgboost
- torch
- ta
- pandas-ta

---

## Estrutura esperada

A estrutura inicial do projeto deve seguir este formato:

```txt
forex-tp-sl-classifier/
│
├── AGENTS.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── eurusd_m5.csv
│   └── processed/
│       └── dataset_eurusd_m5.csv
│
├── models/
│   └── tp_sl_classifier.pkl
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── load_data.py
    ├── features.py
    ├── labeling.py
    ├── train.py
    ├── evaluate.py
    ├── backtest.py
    └── predict.py
```

---

## Formato dos dados de entrada

O CSV original deve conter, no mínimo, as colunas:

```csv
time,open,high,low,close,volume
```

Regras:

- `time` deve ser convertido para `datetime`.
- Os candles devem ser ordenados por `time` em ordem crescente.
- Remover duplicatas por data/hora.
- Validar se `open`, `high`, `low` e `close` são numéricos.
- Não usar dados futuros para criar features.

---

## Regras de rotulagem TP antes do SL

Para cada candle, considerar entrada no fechamento do candle atual.

### Para BUY

```txt
entry = close atual
tp = entry + tp_pips * pip_size
sl = entry - sl_pips * pip_size
```

### Para SELL

```txt
entry = close atual
tp = entry - tp_pips * pip_size
sl = entry + sl_pips * pip_size
```

### Labels

```txt
1  = Take Profit atingido antes do Stop Loss
0  = Stop Loss atingido antes do Take Profit
-1 = Nenhum dos dois foi atingido dentro do limite de candles futuros
```

### Regra conservadora

Se TP e SL forem atingidos no mesmo candle, considerar que o SL foi atingido primeiro.

Motivo: usando apenas dados OHLC, não é possível saber a ordem exata dos eventos dentro do candle. A regra conservadora evita superestimar o resultado do modelo.

---

## Features iniciais

Criar features simples e interpretáveis antes de modelos avançados.

Sugestões:

- `return_1`
- `return_3`
- `return_5`
- `return_10`
- `ma_9`
- `ma_21`
- `ma_50`
- `dist_ma_9`
- `dist_ma_21`
- `dist_ma_50`
- `rsi_14`
- `atr_14`
- `volatility_20`
- `body_size`
- `upper_wick`
- `lower_wick`
- `range_size`
- `hour`
- `day_of_week`

Regras:

- Todas as features devem usar apenas dados do candle atual e dos candles anteriores.
- Não usar `shift(-1)`, dados futuros ou qualquer cálculo que vaze informação do futuro.
- Após criar indicadores com rolling windows, remover linhas com valores nulos.

---

## Separação de treino e teste

Nunca usar split aleatório em séries temporais.

Separar por ordem temporal:

```txt
80% inicial = treino
20% final   = teste
```

Também é recomendado implementar, em uma fase posterior, walk-forward validation.

---

## Treinamento inicial

O script `src/train.py` deve:

1. Carregar o dataset processado.
2. Remover registros com `label = -1`.
3. Separar treino e teste por ordem temporal.
4. Treinar um modelo baseline.
5. Avaliar o modelo.
6. Salvar o modelo em `models/tp_sl_classifier.pkl`.

Modelo inicial sugerido:

- `StandardScaler`
- `MLPClassifier`

Também pode ser criada opção para:

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `LogisticRegression`

---

## Métricas obrigatórias

Não avaliar o modelo somente por acurácia.

Reportar:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Precision da classe positiva
- Matriz de confusão

No backtest, reportar:

- Total de trades
- Win rate
- Lucro/prejuízo em pips
- Profit factor
- Drawdown máximo
- Expectancy por trade
- Sequência máxima de perdas
- Média de pips por trade

---

## Backtest

O arquivo `src/backtest.py` deve:

1. Carregar o modelo treinado.
2. Carregar os dados de teste.
3. Gerar probabilidades.
4. Operar apenas quando a probabilidade for maior ou igual ao threshold definido.
5. Simular resultado com base no label.
6. Considerar TP, SL, spread e custos.
7. Exibir métricas finais.

Exemplo de regra:

```txt
Se probabilidade de TP antes do SL >= 0.70:
    aceitar operação
Caso contrário:
    não operar
```

O threshold deve ser configurável.

---

## Spread e custos

O backtest deve considerar custos de forma conservadora.

Incluir configuração para:

- spread em pips
- slippage em pips
- comissão, se aplicável

O resultado final em pips deve descontar esses custos.

---

## Configurações

Criar `src/config.py` com parâmetros principais:

```python
SYMBOL = "EURUSD"
TIMEFRAME = "M5"

TP_PIPS = 20
SL_PIPS = 15
PIP_SIZE = 0.0001
MAX_BARS_AHEAD = 50

TRAIN_SIZE = 0.80
THRESHOLD = 0.70

SPREAD_PIPS = 1.0
SLIPPAGE_PIPS = 0.2
```

Para pares com JPY, considerar `pip_size = 0.01`.

---

## Comandos esperados

O projeto deve permitir comandos simples:

```bash
python -m src.features
python -m src.train
python -m src.backtest
python -m src.predict
```

Também é aceitável criar scripts com argumentos via CLI posteriormente.

---

## Boas práticas

- Usar funções pequenas.
- Usar nomes claros.
- Evitar código duplicado.
- Usar type hints quando fizer sentido.
- Validar entradas.
- Evitar lógica solta fora de funções.
- Usar `if __name__ == "__main__":` nos scripts executáveis.
- Salvar artefatos em pastas claras.
- Não misturar criação de features, treinamento e backtest no mesmo arquivo.

---

## Cuidados contra vazamento de dados

Este projeto deve ter cuidado extremo com data leakage.

Evitar:

- split aleatório
- indicadores calculados usando candles futuros
- normalização feita antes da separação treino/teste
- uso de labels ou resultados futuros como features
- otimização de threshold diretamente no teste final
- ajustar estratégia olhando o teste repetidamente

O scaler deve ser ajustado apenas no treino e aplicado no teste usando pipeline do scikit-learn.

---

## Walk-forward validation

Em fase posterior, implementar validação walk-forward.

Exemplo:

```txt
Jan-Mar = treino
Abr     = teste

Fev-Abr = treino
Mai     = teste

Mar-Mai = treino
Jun     = teste
```

Gerar relatório com métricas por janela e métricas consolidadas.

---

## Estratégia recomendada

Não deixar a IA decidir tudo sozinha no começo.

Fluxo recomendado:

```txt
Estratégia baseada em regras gera uma possível entrada
↓
Classificador estima probabilidade de TP antes do SL
↓
Se probabilidade >= threshold, operação é aceita
↓
Caso contrário, operação é rejeitada
```

Isso torna o sistema mais controlável e mais fácil de testar.

---

## Fases do projeto

### Fase 1

- Carregar CSV
- Criar features
- Criar labels BUY
- Treinar modelo baseline
- Rodar avaliação simples

### Fase 2

- Criar backtest com threshold
- Considerar spread e slippage
- Gerar relatório de métricas

### Fase 3

- Adicionar SELL
- Comparar BUY e SELL separadamente
- Testar diferentes combinações de TP/SL

### Fase 4

- Implementar walk-forward validation
- Exportar resultados para CSV

### Fase 5

- Testar modelos mais avançados
- LightGBM
- XGBoost
- Redes neurais sequenciais

### Fase 6

- Integração com conta demo
- Nenhuma operação real antes de validação extensa

---

## Aviso importante

Este projeto é experimental e educacional.

Bons resultados em backtest não garantem bons resultados em mercado real. Forex envolve risco elevado, custos operacionais, slippage, mudanças de regime e condições imprevisíveis.

Sempre validar em conta demo antes de qualquer uso prático.
