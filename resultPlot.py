import matplotlib.pyplot as plt
import numpy as np

# Data from the table
categories = ['Line', 'Map', 'Line+Map']
avg_endpoint_error = [0.011, 0.07, 0.019]
percentage_endpoint_error = [3.31, 5.3, 5.71]
avg_point_error = [0.008, 0.07, 0.092]
percentage_point_error = [3.56, 5.3, 7.61]

# X-axis positions for each category
x = np.arange(len(categories))
width = 0.2

# Create a fancy bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - 1.5*width, avg_endpoint_error, width, label='Avg Endpoint Error')
bars2 = ax.bar(x - 0.5*width, percentage_endpoint_error, width, label='% Endpoint Error')
bars3 = ax.bar(x + 0.5*width, avg_point_error, width, label='Avg Point Error')
bars4 = ax.bar(x + 1.5*width, percentage_point_error, width, label='% Point Error')

# Add titles and labels
ax.set_title('Comparison of Errors Across Categories', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Error Values', fontsize=12)
ax.set_xlabel('Categories', fontsize=12)
ax.legend()

# Add grid lines for better visualization
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels on the bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('resultPlot.png')
plt.show()
