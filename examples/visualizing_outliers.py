import numpy as np
import matplotlib.pyplot as plt
from hampel import hampel

original_data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

# Add outliers to the time series
outlier_indices = [20, 40, 60, 80]  # Indices where outliers will be added
outlier_values = [2.0, -1.4, 2.1, -0.5]  # Outlier values corresponding to the indices

for index, value in zip(outlier_indices, outlier_values):
    original_data[index] = value

# Define filter parameters (window_size and n_sigma)
window_size = 5
n_sigma = 3.0

# Apply the Hampel filter
filtered_data = hampel(original_data, window_size, n_sigma)

plt.figure(figsize=(12, 6))

# Plot the original time series
plt.subplot(2, 1, 1)
plt.plot(original_data, label='Original Data', color='blue')
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Plot the filtered time series
plt.subplot(2, 1, 2)
plt.plot(filtered_data, label='Filtered Data', color='green')
plt.title('Filtered Time Series (Without Outliers)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
