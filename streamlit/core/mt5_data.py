"""Conexão e exportação de dataseries OHLCV do terminal MetaTrader 5.

Baseado no padrão de coleta usado nos demais projetos do repositório
(``sma_prediction/data/collector.py``), generalizado para uso interativo
dentro do painel Streamlit: qualquer símbolo/timeframe, por quantidade de
barras ou por intervalo de datas.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

try:  # MetaTrader5 só está disponível no Windows com o terminal instalado
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - ambiente sem MT5
    mt5 = None

_TF_NAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


class MT5Error(RuntimeError):
    """Erro de conexão ou coleta de dados no MT5."""


def is_available() -> bool:
    """Indica se o pacote ``MetaTrader5`` está instalado nesta máquina."""
    return mt5 is not None


def connect() -> dict:
    """Inicializa a conexão com o terminal MT5 e retorna dados da conta.

    Returns:
        Dicionário com informações da conta conectada (login, servidor,
        empresa, saldo, moeda).

    Raises:
        MT5Error: se o pacote não estiver instalado ou a inicialização falhar.
    """
    if mt5 is None:
        raise MT5Error(
            "Pacote 'MetaTrader5' não está instalado nesta máquina. "
            "Rode: pip install MetaTrader5"
        )
    if not mt5.initialize():
        code, msg = mt5.last_error()
        raise MT5Error(
            f"Falha ao inicializar o MT5 ({code}: {msg}). "
            "Verifique se o terminal MetaTrader 5 está aberto e logado."
        )
    account = mt5.account_info()
    if account is None:
        return {"conectado": True, "conta": None}
    return {
        "conectado": True,
        "login": account.login,
        "servidor": account.server,
        "empresa": account.company,
        "saldo": account.balance,
        "moeda": account.currency,
    }


def disconnect() -> None:
    """Encerra a conexão com o terminal MT5, se estiver aberta."""
    if mt5 is not None:
        mt5.shutdown()


def _resolve_timeframe(timeframe: str) -> int:
    """Converte o timeframe textual (ex.: ``"H1"``) na constante do MT5."""
    key = timeframe.upper()
    if key not in _TF_NAMES:
        raise ValueError(
            f"Timeframe inválido: '{timeframe}'. Use um de: {', '.join(_TF_NAMES)}"
        )
    return getattr(mt5, _TF_NAMES[key])


def validate_symbol(symbol: str) -> None:
    """Verifica se o símbolo existe no MT5 e o torna visível no Market Watch.

    Raises:
        MT5Error: se o símbolo não existir ou não puder ser selecionado.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        prefixo = symbol[:3].upper()
        todos = mt5.symbols_get() or []
        similares = [s.name for s in todos if s.name.upper().startswith(prefixo)][:10]
        sugestao = f" Símbolos similares: {similares}" if similares else ""
        raise MT5Error(f"Símbolo '{symbol}' não encontrado no MT5.{sugestao}")
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise MT5Error(f"Não foi possível selecionar '{symbol}' no Market Watch.")


def search_symbols(termo: str, limite: int = 30) -> list[str]:
    """Busca símbolos disponíveis no MT5 cujo nome contenha ``termo``.

    Requer conexão já estabelecida (``connect()``).
    """
    if mt5 is None:
        return []
    todos = mt5.symbols_get() or []
    termo = termo.upper().strip()
    if not termo:
        return [s.name for s in todos[:limite]]
    return [s.name for s in todos if termo in s.name.upper()][:limite]


def export_ohlcv(
    symbol: str,
    timeframe: str,
    modo: str = "barras",
    bars: int = 5000,
    data_inicio: dt.date | None = None,
    data_fim: dt.date | None = None,
) -> pd.DataFrame:
    """Exporta uma dataseries OHLCV do MT5.

    Args:
        symbol: nome do ativo (ex.: "EURUSD").
        timeframe: código do timeframe (ex.: "H1").
        modo: ``"barras"`` para baixar as últimas ``bars`` barras fechadas,
            ou ``"periodo"`` para baixar entre ``data_inicio`` e ``data_fim``.
        bars: quantidade de barras (usado quando ``modo == "barras"``).
        data_inicio: data inicial (usado quando ``modo == "periodo"``).
        data_fim: data final (usado quando ``modo == "periodo"``).

    Returns:
        DataFrame com colunas ``time, open, high, low, close, tick_volume,
        spread, real_volume``, apenas barras fechadas e com volume > 0.

    Raises:
        MT5Error: em qualquer falha de conexão, símbolo ou coleta.
    """
    if mt5 is None:
        raise MT5Error("Pacote 'MetaTrader5' não instalado.")

    connect()
    try:
        validate_symbol(symbol)
        tf = _resolve_timeframe(timeframe)

        if modo == "periodo":
            if data_inicio is None or data_fim is None:
                raise ValueError("Informe data_inicio e data_fim para modo='periodo'.")
            rates = mt5.copy_rates_range(
                symbol,
                tf,
                dt.datetime.combine(data_inicio, dt.time.min),
                dt.datetime.combine(data_fim, dt.time.max),
            )
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise MT5Error(f"Nenhuma barra retornada para {symbol} {timeframe} ({code}: {msg}).")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        antes = len(df)
        df = df[df["tick_volume"] > 0].reset_index(drop=True)
        removidas = antes - len(df)

        # A última barra pode estar em formação; descarta para usar só fechadas
        if modo == "barras" and len(df) > 0:
            df = df.iloc[:-1].reset_index(drop=True)

        if len(df) < 50:
            raise MT5Error(
                f"Barras insuficientes após limpeza: {len(df)}. "
                "Aumente o período ou escolha outro timeframe."
            )

        df.attrs["symbol"] = symbol
        df.attrs["timeframe"] = timeframe
        df.attrs["removidas"] = removidas
        return df
    finally:
        disconnect()
