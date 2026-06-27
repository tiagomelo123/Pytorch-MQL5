from src.config import (
    dataset_key,
    model_path,
    pip_size_for_symbol,
    processed_data_path,
    raw_data_path,
)


def test_dynamic_paths_include_symbol_and_timeframe():
    assert dataset_key("gbpusd", "m15") == "gbpusd_m15"
    assert raw_data_path("GBPUSD", "M15").name == "gbpusd_m15.csv"
    assert processed_data_path("GBPUSD", "M15").name == "dataset_gbpusd_m15.csv"
    assert model_path("GBPUSD", "M15").name == "tp_sl_classifier_gbpusd_m15.pkl"


def test_jpy_pairs_use_jpy_pip_size():
    assert pip_size_for_symbol("USDJPY") == 0.01
    assert pip_size_for_symbol("EURUSD") == 0.0001
