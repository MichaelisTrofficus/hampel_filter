import numpy as np
cimport numpy as np

def hampel_filter(np.ndarray[np.float64_t, ndim=1] data, int window_size=3, float n_sigma=3.0):
    """
    Applies the Hampel filter to a 1-dimensional numpy array for outlier detection.

    This function replaces outliers in the input data with the median value within a moving window.

    Parameters:
        data (numpy.ndarray): The input 1-dimensional numpy array to be filtered.
        window_size (int, optional): The size of the moving window for outlier detection. Default is 3.
        n_sigma (float, optional): The number of standard deviations for outlier detection. Default is 3.0.

    Returns:
        numpy.ndarray: A copy of the input data with outliers replaced by median values within the specified window.

    Example:
        >>> import numpy as np
        >>> from hampel_filter import hampel_filter
        >>> data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
        >>> filtered_data = hampel_filter(data, window_size=3, n_sigma=3.0)
        >>> print(filtered_data)
        [1. 2. 3. 4. 4. 5. 6.]
    """
    cdef int data_len = data.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] filtered_data = data.copy()
    cdef int half_window = window_size // 2

    for i in range(half_window, data_len - half_window):
        window = data[i - half_window: i + half_window + 1]
        median = np.median(window)
        median_absolute_deviation = np.median(np.abs(window - median))
        threshold = n_sigma * 1.4826 * median_absolute_deviation

        if abs(data[i] - median) > threshold:
            filtered_data[i] = median

    return filtered_data
