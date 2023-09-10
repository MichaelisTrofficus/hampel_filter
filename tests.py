import numpy as np
import pandas as pd
import pytest
from hampel import hampel


def test_hampel_filter_no_outliers():
    # Test when there are no outliers, data should remain unchanged
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered_data = hampel(data)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_with_outliers():
    # Test with outliers, they should be replaced by medians within the window
    data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
    filtered_data = hampel(data, window_size=3, n_sigma=3.0)
    expected_result = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    assert np.allclose(expected_result, filtered_data)


def test_hampel_filter_large_window():
    # Test with a large window, should have no effect as the window is too large
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered_data = hampel(data, window_size=10, n_sigma=3.0)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_custom_threshold():
    # Test with a custom threshold, should replace the outlier
    data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
    filtered_data = hampel(data, window_size=3, n_sigma=5.0)
    expected_result = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    assert np.allclose(expected_result, filtered_data)


def test_hampel_filter_empty_data():
    # Test with an empty data array, should return an empty array
    data = np.array([])
    filtered_data = hampel(data)
    assert len(filtered_data) == 0


def test_hampel_filter_one_point():
    # Test with one data point, should return the same point
    data = np.array([42.0])
    filtered_data = hampel(data)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_two_points():
    # Test with two data points, should return the same points
    data = np.array([1.0, 2.0])
    filtered_data = hampel(data)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_three_points():
    # Test with three data points, should return the same points
    data = np.array([2.0, 25.0, 6.0])
    expected_data = np.array([2.0, 6.0, 6.0])
    filtered_data = hampel(data)
    assert np.allclose(expected_data, filtered_data)


def test_hampel_filter_three_points_with_outliers():
    # Test with three data points and outliers, should replace the outliers
    data = np.array([1.0, 100.0, 3.0])
    filtered_data = hampel(data, window_size=3, n_sigma=3.0)
    expected_result = np.array([1.0, 3.0, 3.0])
    assert np.allclose(expected_result, filtered_data)


@pytest.fixture
def sample_dataframe():
    data = {'A': [1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0],
            'B': [10.0, 20.0, 30.0, 200.0, 40.0, 50.0, 60.0]}
    return pd.DataFrame(data)


def test_hampel_filter_dataframe(sample_dataframe):
    filtered_dataframe = sample_dataframe.apply(hampel, axis=0)

    # Define the expected filtered values for each column
    expected_filtered_values = {
        'A': np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]),
        'B': np.array([10.0, 20.0, 30.0, 40.0, 40.0, 50.0, 60.0])
    }

    for column in sample_dataframe.columns:
        assert np.allclose(filtered_dataframe[column], expected_filtered_values[column])


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