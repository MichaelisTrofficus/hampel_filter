import numpy as np
from dataclasses import dataclass


@dataclass
class Result:
    filtered_data: np.ndarray
    outlier_indices: np.ndarray
    medians: np.ndarray
    median_absolute_deviations: np.ndarray
    thresholds: np.ndarray
