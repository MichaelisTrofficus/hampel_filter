import pandas as pd
import pytest

from hampel import hampel


@pytest.fixture
def ts_data():
    return pd.Series([1, 2, 1, 1, 40, 2, 1, 1, 30, 40, 1, 1, 2, 1])


def test_str_ts():
    with pytest.raises(ValueError):
        hampel("a", -1, 3)


def test_negative_window_size(ts_data):
    with pytest.raises(ValueError):
        hampel(ts_data, -1, 3)


def test_zero_window_size(ts_data):
    with pytest.raises(ValueError):
        hampel(ts_data, 0, 3)


def test_str_window_key(ts_data):
    with pytest.raises(ValueError):
        hampel(ts_data, "a", 3)


def test_negative_sigma(ts_data):
    with pytest.raises(ValueError):
        hampel(ts_data, 3, -1)


def test_str_sigma(ts_data):
    with pytest.raises(ValueError):
        hampel(ts_data, 1, "a")