import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.features import add_features


def test_add_features_creates_expected_columns_without_future_rows():
    rows = 80
    close = 1.10 + np.arange(rows) * 0.0001
    df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=rows, freq="5min"),
            "open": close - 0.00005,
            "high": close + 0.00020,
            "low": close - 0.00020,
            "close": close,
            "volume": np.arange(rows) + 100,
        }
    )

    featured = add_features(df).dropna()

    assert set(FEATURE_COLUMNS).issubset(featured.columns)
    assert len(featured) < len(df)
    assert not featured[FEATURE_COLUMNS].isna().any().any()
