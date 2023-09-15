from functools import wraps
from typing import Union

import numpy as np
import pandas as pd


def type_check_decorator(func):
    @wraps(func)
    def wrapper(data: Union[np.ndarray, pd.Series], window_size: int = 5, n_sigma: float = 3.0):
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
                raise ValueError("Threshold must be equal or more than 0.")

        return func(data, window_size, n_sigma)

    return wrapper
