# Forex TP/SL Classifier

MVP em Python para estimar a probabilidade de uma entrada BUY atingir o Take Profit antes do Stop Loss.

O projeto usa features historicas do candle atual e anteriores, cria labels conservadores de TP antes do SL e treina uma rede neural simples com `MLPClassifier`.

## Estrutura

```txt
data/raw/eurusd_m5.csv
data/processed/dataset_eurusd_m5_tp20_sl15.csv
models/tp_sl_classifier_eurusd_m5_tp20_sl15.pkl
reports/eurusd_m5_tp20_sl15/metrics.json
reports/eurusd_m5_tp20_sl15/backtest_metrics.json
reports/eurusd_m5_tp20_sl15/learning_curve.png
src/
```

## Dados de entrada

Coloque o CSV original em:

```txt
data/raw/eurusd_m5.csv
```

Colunas obrigatorias:

```csv
time,open,high,low,close,volume
```

## Comandos

Pipeline completo:

```bash
python -m src.pipeline --symbol EURUSD --timeframe M5 --tp-pips 20 --sl-pips 15 --bars 5000 --threshold 0.70
```

Usar pipeline com CSV ja existente, sem importar do MT5:

```bash
python -m src.pipeline --symbol EURUSD --timeframe M5 --tp-pips 20 --sl-pips 15 --skip-import
```

Importar candles direto do MetaTrader 5:

```bash
python -m src.mt5_import --symbol EURUSD --timeframe M5 --bars 5000
```

Criar dataset processado com features e labels:

```bash
python -m src.features --symbol EURUSD --timeframe M5 --tp-pips 20 --sl-pips 15
```

Treinar a rede neural:

```bash
python -m src.train --symbol EURUSD --timeframe M5 --tp-pips 20 --sl-pips 15
```

O treino salva automaticamente:

- modelo em `models/tp_sl_classifier_<ativo>_<timeframe>_tp<TP>_sl<SL>.pkl`
- metricas em `reports/<ativo>_<timeframe>_tp<TP>_sl<SL>/metrics.json`
- grafico de aprendizagem em `reports/<ativo>_<timeframe>_tp<TP>_sl<SL>/learning_curve.png`

Rodar backtest no trecho final da serie:

```bash
python -m src.backtest --symbol EURUSD --timeframe M5 --tp-pips 20 --sl-pips 15 --threshold 0.70
```

O backtest tambem salva `reports/<ativo>_<timeframe>_tp<TP>_sl<SL>/backtest_metrics.json`.

Gerar probabilidade para o candle mais recente:

```bash
python -m src.predict --symbol EURUSD --timeframe M5 --tp-pips 20 --sl-pips 15
```

Rodar testes automatizados:

```bash
pytest
```

Para outro ativo/timeframe:

```bash
python -m src.mt5_import --symbol GBPUSD --timeframe M15 --bars 10000
python -m src.features --symbol GBPUSD --timeframe M15 --tp-pips 15 --sl-pips 10
python -m src.train --symbol GBPUSD --timeframe M15 --tp-pips 15 --sl-pips 10
python -m src.backtest --symbol GBPUSD --timeframe M15 --tp-pips 15 --sl-pips 10
```

## Modelo inicial

O baseline e uma rede neural do scikit-learn:

- `StandardScaler`
- `MLPClassifier`

O split e temporal: 80% inicial para treino e 20% final para teste. Registros com `label = -1` sao removidos no treino, pois representam casos em que nem TP nem SL foram atingidos dentro do horizonte configurado.

## Aviso

Projeto experimental e educacional. Resultados de backtest nao garantem resultados em mercado real.
