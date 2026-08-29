# Painel de Redes Neurais — MetaTrader 5 + PyTorch + Firebase

Painel em Streamlit para exportar dataseries do MetaTrader 5, treinar e
comparar redes neurais em PyTorch, com armazenamento dos datasets e modelos
no Firebase Storage.

## Instalação

```bash
cd streamlit
pip install -r requirements.txt
```

Requisitos:

- Windows com o terminal **MetaTrader 5** instalado e logado em uma conta
  (necessário apenas para a exportação de dados — o restante do painel
  funciona sem o MT5, a partir de datasets já em cache).
- (Opcional) Projeto no **Firebase** com Storage habilitado, se quiser
  guardar datasets e modelos na nuvem.

## Configurar o Firebase (opcional)

1. No Console do Firebase, gere uma chave de service account (JSON).
2. Copie `.streamlit/secrets.toml.example` para `.streamlit/secrets.toml`.
3. Preencha `credentials_path` (caminho do JSON) e `storage_bucket`.

Sem essa configuração, o painel funciona normalmente usando apenas o cache
local em `data_cache/` e `runs/`.

## Rodar o painel

```bash
streamlit run app.py
```

## Estrutura

```
streamlit/
  app.py                        # página inicial / status das conexões
  pages/
    1_📥_Exportar_Dados.py       # conecta ao MT5 e exporta dataseries
    2_🗂️_Datasets.py             # gerencia datasets (local + Firebase)
    3_🧠_Treinar_Modelo.py       # configura e treina o modelo, com progresso ao vivo
    4_📊_Comparar_Modelos.py     # compara métricas e curvas de perda entre runs
  core/
    mt5_data.py                  # conexão e coleta OHLCV do MT5
    firebase_client.py           # upload/download no Firebase Storage
    features.py                  # engenharia de features (retornos, médias, RSI, ...)
    labeling.py                  # rotulagem de pullback vs. continuação de tendência
    dataset.py                   # janelas supervisionadas, split, scaler, DataLoaders
    models.py                    # arquiteturas LSTM / GRU / MLP
    train.py                     # loop de treino com early stopping + avaliação
    registry.py                  # registro local dos modelos treinados (runs/registry.json)
  data_cache/                    # datasets baixados do MT5 (CSV)
  runs/                          # artefatos dos modelos treinados (model.pt, métricas, ...)
```

## Fluxo de uso

1. **Exportar Dados**: escolha símbolo, timeframe e período/quantidade de
   barras, conecte ao MT5 e exporte. Salve em cache local e, se quiser,
   envie para o Firebase Storage.
2. **Datasets**: veja os datasets em cache local e no Firebase, baixe ou
   remova conforme necessário.
3. **Treinar Modelo**: escolha um dataset, o tipo de tarefa, as features de
   entrada, a arquitetura (LSTM/GRU/MLP) e os hiperparâmetros. Acompanhe a
   curva de perda e as métricas em tempo real durante o treino. Tarefas
   disponíveis:
   - **Regressão (retorno futuro)** — prevê o retorno N barras à frente.
   - **Classificação (direção do preço)** — prevê se o preço sobe ou desce
     N barras à frente.
   - **Classificação (pullback vs. continuação de tendência)** — detecta a
     tendência vigente (EMA rápida x lenta), identifica retrações
     (pullbacks) contra essa tendência a partir do último topo/fundo
     (swing), e classifica se o pullback tende a **continuar** a tendência
     (rompe o extremo do swing a favor da tendência) ou **reverter/falhar**
     (rompe a estrutura contrária antes). Parâmetros ajustáveis: períodos
     das EMAs, barras para confirmar um swing, horizonte de checagem da
     continuação e retração mínima para considerar um pullback.
4. **Comparar Modelos**: veja todos os modelos já treinados em uma tabela,
   compare a métrica principal entre eles e sobreponha curvas de perda de
   validação de até 5 runs.

## Extensível

Novas features podem ser adicionadas em `core/features.py`
(`FEATURES_DISPONIVEIS`), e novas arquiteturas em `core/models.py`
(`build_model`) — ambas aparecem automaticamente nos seletores da página de
treino.
