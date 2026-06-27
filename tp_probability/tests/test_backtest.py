import pandas as pd

from src.backtest import calculate_backtest_metrics, max_losing_streak


def test_backtest_metrics_calculate_profit_factor_and_drawdown():
    trades = pd.DataFrame({"pips": [10.0, -5.0, -5.0, 20.0]})
    trades["equity"] = trades["pips"].cumsum()

    metrics = calculate_backtest_metrics(trades)

    assert metrics["total_trades"] == 4
    assert metrics["win_rate"] == 0.5
    assert metrics["net_pips"] == 20.0
    assert metrics["profit_factor"] == 3.0
    assert metrics["max_drawdown"] == 10.0
    assert metrics["max_losing_streak"] == 2


def test_max_losing_streak_handles_no_losses():
    assert max_losing_streak(pd.Series([1.0, 2.0, 3.0])) == 0
