from typing import Union

import numpy as np
import pandas as pd
from hampel.extension.hampel import hampel as hampel_extension

from hampel.decorator import type_check_decorator
from hampel.result import Result


@type_check_decorator
def hampel(data: Union[np.ndarray, pd.Series], window_size: int = 5, n_sigma: float = 3.0) -> Result:
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
    is_pd_series = False

    if isinstance(data, pd.Series):
        data = data.copy().to_numpy()
        is_pd_series = True

    result = hampel_extension(np.asarray(data, dtype=np.float32), window_size, n_sigma)

    return Result(
        filtered_data=result[0] if not is_pd_series else pd.Series(result[0]),
        outlier_indices=result[1],
        medians=result[2],
        median_absolute_deviations=result[3],
        thresholds=result[4],
    )
