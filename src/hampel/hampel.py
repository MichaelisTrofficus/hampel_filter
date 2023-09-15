from typing import Union

import numpy as np
import pandas as pd
from hampel.extension.hampel import hampel as hampel_extension


def hampel(data: Union[np.ndarray, pd.Series], window_size: int = 5, n_sigma: float = 3.0):
    """
    Apply the Hampel filter for outlier detection to a pandas.Series or numpy.ndarray.

    The Hampel filter identifies and replaces outliers in the input data with the median
    value within a moving window.

    Parameters:
        data (pandas.Series or numpy.ndarray): The input 1-dimensional data to be filtered.
        window_size (int, optional): The size of the moving window for outlier detection (default is 5).
        n_sigma (float, optional): The number of standard deviations for outlier detection (default is 3.0).

    Returns:
        numpy.ndarray: A copy of the input data with outliers replaced by median values within the specified window.

    Raises:
        ValueError: If input data is not a pandas.Series or numpy.ndarray.
        ValueError: If window_size is not an integer or is less than or equal to 0.
        ValueError: If n_sigma is not a float or is less than 0.

    Example:
        >>> import pandas as pd
        >>> from hampel import hampel
        >>> data = pd.Series([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
        >>> filtered_data = hampel(data, window_size=3, n_sigma=3.0)
        >>> print(filtered_data)
        [1. 2. 3. 4. 4. 5. 6.]

    Note:
        - If the input data is a pandas.Series, it will be converted to a numpy.ndarray for processing.
    """

    if not (isinstance(data, pd.Series) or isinstance(data, np.ndarray)):
        raise ValueError("Input data must be a pandas.Series or a numpy.ndarray")

    if type(window_size) != int:
        raise ValueError("Window size must be of type integer.")
    else:
        if window_size <= 0:
            raise ValueError("Window size must be more than 0.")

    if type(n_sigma) != float:
        raise ValueError("Threshold must be of type float.")
    else:
        if n_sigma < 0:
            raise ValueError("Window size must be equal or more than 0.")

    if isinstance(data, pd.Series):
        data = data.copy().to_numpy()

    return hampel_extension(np.asarray(data, dtype=np.float32), window_size, n_sigma)
