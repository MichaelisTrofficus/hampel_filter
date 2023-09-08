import numpy as np
from hampel_filter import hampel_filter


def test_hampel_filter_no_outliers():
    # Test when there are no outliers, data should remain unchanged
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered_data = hampel_filter(data)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_with_outliers():
    # Test with outliers, they should be replaced by medians within the window
    data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
    filtered_data = hampel_filter(data, window_size=3, n_sigma=3.0)
    expected_result = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    assert np.allclose(expected_result, filtered_data)


def test_hampel_filter_large_window():
    # Test with a large window, should have no effect as the window is too large
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered_data = hampel_filter(data, window_size=10, n_sigma=3.0)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_custom_threshold():
    # Test with a custom threshold, should replace the outlier
    data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
    filtered_data = hampel_filter(data, window_size=3, n_sigma=5.0)
    expected_result = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    assert np.allclose(expected_result, filtered_data)


def test_hampel_filter_empty_data():
    # Test with an empty data array, should return an empty array
    data = np.array([])
    filtered_data = hampel_filter(data)
    assert len(filtered_data) == 0


def test_hampel_filter_one_point():
    # Test with one data point, should return the same point
    data = np.array([42.0])
    filtered_data = hampel_filter(data)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_two_points():
    # Test with two data points, should return the same points
    data = np.array([1.0, 2.0])
    filtered_data = hampel_filter(data)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_three_points():
    # Test with three data points, should return the same points
    data = np.array([1.0, 2.0, 3.0])
    filtered_data = hampel_filter(data)
    assert np.allclose(data, filtered_data)


def test_hampel_filter_three_points_with_outliers():
    # Test with three data points and outliers, should replace the outliers
    data = np.array([1.0, 100.0, 3.0])
    filtered_data = hampel_filter(data, window_size=3, n_sigma=3.0)
    expected_result = np.array([1.0, 3.0, 3.0])
    assert np.allclose(expected_result, filtered_data)
