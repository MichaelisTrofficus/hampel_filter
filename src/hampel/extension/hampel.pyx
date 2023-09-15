import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hampel(np.ndarray[np.float32_t, ndim=1] data, int window_size, float n_sigma):
    """
    Applies the Hampel filter to a 1-dimensional numpy array for outlier detection.

    This function replaces outliers in the input data with the median value within a moving window.

    Parameters:
        data (numpy.ndarray): The input 1-dimensional numpy array to be filtered.
        window_size (int, optional): The size of the moving window for outlier detection.
        n_sigma (float, optional): The number of standard deviations for outlier detection.

    Returns:
        Tuple[numpy.ndarray, List[int], numpy.ndarray, numpy.ndarray]: A tuple containing:
            - Filtered data (numpy.ndarray): A copy of the input data with outliers replaced by median values within the specified window.
            - Indices of outliers (List[int]): A list of indices where outliers were found.
            - Local medians (numpy.ndarray): An array containing local median values for each element of the input data.
            - Estimated standard deviations (numpy.ndarray): An array containing estimated standard deviations for each element of the input data.
    """
    cdef:
        int i, j, window_length
        np.ndarray[np.float32_t, ndim=1] window
        float median, median_absolute_deviation, threshold

    cdef int data_len = data.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] filtered_data = data.copy()
    cdef int half_window = window_size // 2

    # Preallocate memory for threshold and outlier_indices arrays
    cdef np.ndarray[np.float32_t, ndim=1] thresholds = np.empty(data_len, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] outlier_indices = np.empty(data_len, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] median_absolute_deviations = np.empty(data_len, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] medians = np.empty(data_len, dtype=np.float32)

    cdef int num_outliers = 0

    for i in range(half_window, data_len - half_window):
        window = data[i - half_window: i + half_window + 1].copy()
        window_length = len(window)
        median = np.median(window)

        for j in range(window_length):
            window[j] = np.abs(window[j] - median)

        median_absolute_deviation = np.median(window)
        threshold = n_sigma * 1.4826 * median_absolute_deviation
        thresholds[i] = threshold
        median_absolute_deviations[i] = median_absolute_deviation
        medians[i] = median

        if abs(data[i] - median) > threshold:
            filtered_data[i] = median
            outlier_indices[num_outliers] = i
            num_outliers += 1

    # Trim the outlier_indices array to its actual size
    outlier_indices = outlier_indices[:num_outliers]

    return (
        filtered_data,
        outlier_indices,
        medians,
        median_absolute_deviations,
        thresholds
    )
