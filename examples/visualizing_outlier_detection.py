import matplotlib.pyplot as plt
import numpy as np
from hampel import hampel


original_data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

# Add outliers to the original data
for index, value in zip([20, 40, 60, 80], [2.0, -1.4, 2.1, -0.5]):
    original_data[index] = value

result = hampel(original_data)

filtered_data = result.filtered_data
outlier_indices = result.outlier_indices
medians = result.medians
thresholds = result.thresholds

fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Plot the original data with estimated standard deviations in the first subplot
axes[0].plot(original_data, label='Original Data', color='b')
axes[0].fill_between(range(len(original_data)), medians + thresholds,
                     medians - thresholds, color='gray', alpha=0.5, label='Median +- Threshold')
axes[0].set_xlabel('Data Point')
axes[0].set_ylabel('Value')
axes[0].set_title('Original Data with Bands representing Upper and Lower limits')

for i in outlier_indices:
    axes[0].plot(i, original_data[i], 'ro', markersize=5)  # Mark as red

axes[0].legend()

# Plot the filtered data in the second subplot
axes[1].plot(filtered_data, label='Filtered Data', color='g')
axes[1].set_xlabel('Data Point')
axes[1].set_ylabel('Value')
axes[1].set_title('Filtered Data')
axes[1].legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()
