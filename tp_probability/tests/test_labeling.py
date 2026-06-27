import pandas as pd

from src.labeling import create_buy_labels


def test_buy_label_marks_tp_before_sl():
    df = pd.DataFrame(
        {
            "close": [1.1000, 1.1005, 1.1021],
            "high": [1.1002, 1.1010, 1.1022],
            "low": [1.0998, 1.1000, 1.1015],
        }
    )

    labels = create_buy_labels(df, tp_pips=20, sl_pips=15, pip_size=0.0001, max_bars_ahead=2)

    assert labels.iloc[0] == 1


def test_buy_label_uses_conservative_sl_when_tp_and_sl_same_candle():
    df = pd.DataFrame(
        {
            "close": [1.1000, 1.1000],
            "high": [1.1001, 1.1021],
            "low": [1.0999, 1.0984],
        }
    )

    labels = create_buy_labels(df, tp_pips=20, sl_pips=15, pip_size=0.0001, max_bars_ahead=1)

    assert labels.iloc[0] == 0


def test_buy_label_marks_minus_one_when_no_target_hit():
    df = pd.DataFrame(
        {
            "close": [1.1000, 1.1002, 1.1001],
            "high": [1.1001, 1.1004, 1.1003],
            "low": [1.0999, 1.0998, 1.0999],
        }
    )

    labels = create_buy_labels(df, tp_pips=20, sl_pips=15, pip_size=0.0001, max_bars_ahead=2)

    assert labels.iloc[0] == -1
