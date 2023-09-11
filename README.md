# Hampel

![hampel_usage.png](img%2Fhampel_usage.png)


The Hampel filter is generally used to detect anomalies in data with a timeseries structure.
It basically consists of a sliding window of a parameterizable size. 
For each window, each observation will be compared with the Median Absolute Deviation (MAD).
The observation will be considered an outlier in the case in which it exceeds the MAD by n times (the parameter n is also parameterizable).

For more details, see the [Related Links](#related-links) section.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [License](#license)
- [Contributing](#contributing)
- [Related Links](#related-links)


## Installation

To use the Hampel filter in your Python project, you can install it via pip:

```
pip install hampel
```

## Usage

Here's how you can use the Hampel filter in your Python code:

```python
import pandas as pd
from hampel import hampel

# Sample data as a pandas.Series
data = pd.Series([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])

# Apply the Hampel filter
filtered_data = hampel(data, window_size=3, n_sigma=3.0)

print(filtered_data)
```

### Applying the Hampel Filter to a Pandas DataFrame

If you want to directly apply hampel filter to multiple columns in a  `pandas.Dataframe`,
follow this code:

```python
import pandas as pd
from hampel import hampel

df = pd.DataFrame({
    'A': [1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0],
    'B': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
})

filtered_df = df.apply(hampel, axis=0)

print(df)
```

## Parameters

* `data`: The input 1-dimensional data to be filtered (pandas.Series or numpy.ndarray).
* `window_size` (optional): The size of the moving window for outlier detection (default is 5).
* `n_sigma` (optional): The number of standard deviations for outlier detection (default is 3.0).

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or suggest improvements.

## Related Links

https://medium.com/wwblog/clean-up-your-time-series-data-with-a-hampel-filter-58b0bb3ebb04

https://en.wikipedia.org/wiki/Median_absolute_deviation
