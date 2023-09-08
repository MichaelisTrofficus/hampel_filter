import pandas as pd
from hampel_filter import hampel_filter


def hampel(data, window_size=3, n_sigma=3.0):
    if isinstance(data, pd.Series):
        data = data.copy().to_numpy()
    return hampel_filter(data, window_size, n_sigma)
