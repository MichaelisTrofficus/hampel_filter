from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


@dataclass
class Result:
    filtered_data: Union[np.ndarray, pd.Series]
    outlier_indices: np.ndarray
    medians: np.ndarray
    median_absolute_deviations: np.ndarray
    thresholds: np.ndarray
