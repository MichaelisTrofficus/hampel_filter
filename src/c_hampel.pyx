import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.cdivision(True)
def _hampel(np.ndarray[np.float32_t, ndim=1] data, int window_size=5, float n_sigma=3.0):
    """
    Applies the Hampel filter to a 1-dimensional numpy array for outlier detection.

    This function replaces outliers in the input data with the median value within a moving window.

    Parameters:
        data (numpy.ndarray): The input 1-dimensional numpy array to be filtered.
        window_size (int, optional): The size of the moving window for outlier detection. Default is 3.
        n_sigma (float, optional): The number of standard deviations for outlier detection. Default is 3.0.

    Returns:
        numpy.ndarray: A copy of the input data with outliers replaced by median values within the specified window.
    """
    cdef:
        int i, j, window_length
        np.ndarray[np.float32_t, ndim=1] window
        float median, median_absolute_deviation, threshold

    cdef int data_len = data.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] filtered_data = data.copy()
    cdef int half_window = window_size // 2

    for i in range(half_window, data_len - half_window):
        window = data[i - half_window: i + half_window + 1].copy()
        window_length = len(window)
        median = np.median(window)

        for j in range(window_length):
            window[j] = np.abs(window[j] - median)

        median_absolute_deviation = np.median(window)
        threshold = n_sigma * 1.4826 * median_absolute_deviation

        if abs(data[i] - median) > threshold:
            filtered_data[i] = median

    return np.asarray(filtered_data, dtype=np.float32)
