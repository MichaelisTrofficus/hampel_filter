import numpy as np
import pandas as pd
from _hampel import hampel_filter


def hampel(data, window_size=3, n_sigma=3.0):

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

    return hampel_filter(np.asarray(data, dtype=np.float32), window_size, n_sigma)
