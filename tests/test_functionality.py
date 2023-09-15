import numpy as np
import pandas as pd
import pytest

from hampel import hampel


@pytest.fixture
def sample_dataframe():
    data = {'A': [1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0],
            'B': [10.0, 20.0, 30.0, 200.0, 40.0, 50.0, 60.0]}
    return pd.DataFrame(data)


def test_hampel_filter_no_outliers():
    # Test when there are no outliers, data should remain unchanged
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = hampel(data)
    assert np.allclose(data, result.filtered_data)


def test_hampel_filter_with_outliers():
    # Test with outliers, they should be replaced by medians within the window
    data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
    filtered_data = hampel(data, window_size=5, n_sigma=3.0).filtered_data
    expected_result = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    assert np.allclose(expected_result, filtered_data)


def test_hampel_filter_large_window():
    # Test with a large window, should have no effect as the window is too large
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered_data = hampel(data, window_size=10, n_sigma=3.0).filtered_data
    assert np.allclose(data, filtered_data)


def test_hampel_filter_custom_threshold():
    # Test with a custom threshold, should replace the outlier
    data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
    filtered_data = hampel(data, window_size=5, n_sigma=5.0).filtered_data
    expected_result = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    assert np.allclose(expected_result, filtered_data)


def test_hampel_filter_empty_data():
    # Test with an empty data array, should return an empty array
    data = np.array([])
    filtered_data = hampel(data).filtered_data
    assert len(filtered_data) == 0


def test_hampel_filter_one_point():
    # Test with one data point, should return the same point
    data = np.array([42.0])
    filtered_data = hampel(data).filtered_data
    assert np.allclose(data, filtered_data)


def test_hampel_filter_three_points_with_outliers():
    # Test with three data points and outliers, should replace the outliers
    data = np.array([1.0, 100.0, 3.0])
    filtered_data = hampel(data, window_size=3, n_sigma=3.0).filtered_data
    expected_result = np.array([1.0, 3.0, 3.0])
    assert np.allclose(expected_result, filtered_data)


def test_hampel_filter_dataframe(sample_dataframe):
    filtered_dataframe = sample_dataframe.apply(lambda x: hampel(x).filtered_data, axis=0)

    # Define the expected filtered values for each column
    expected_filtered_values = {
        'A': np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]),
        'B': np.array([10.0, 20.0, 30.0, 40.0, 40.0, 50.0, 60.0])
    }

    for column in sample_dataframe.columns:
        assert np.allclose(filtered_dataframe[column], expected_filtered_values[column])
