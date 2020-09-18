# Theory

The Hampel filter is generally used to detect anomalies in data with a timeseries structure. It basically consists of a sliding window of a parameterizable size. For each window, each observation will be compared with the Median Absolute Deviation (MAD). The observation will be considered an outlier in the case in which it exceeds the MAD by n times (the parameter n is also parameterizable). For more details, see the following Medium post as well as the Wikipedia entry on MAD.

https://medium.com/wwblog/clean-up-your-time-series-data-with-a-hampel-filter-58b0bb3ebb04

https://en.wikipedia.org/wiki/Median_absolute_deviation

# Install Package

To install the package execute the following command.

```
pip install hampel
```

# Usage

```
hampel(ts, window_size=5, n=3)
```

# Arguments

- **ts** - A pandas Series object representing the timeseries 
- **window_size** -  Total window size will be computed as 2*window_size + 1
- **n** - Threshold, default is 3 (Pearson's rule)


# Code Example

```
import matplotlib.pyplot as plt
import pandas as pd
from hampel import hampel

ts = pd.Series([1, 2, 1 , 1 , 1, 2, 13, 2, 1, 2, 15, 1, 2])
ts_cleaned = hampel(ts, window_size=5, n=3)

ts.plot(style="k-")
ts_cleaned.plot(style="g-")
plt.show()
```

