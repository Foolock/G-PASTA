

# Sample data

import matplotlib.pyplot as plt
import numpy as np

des_perf_task_runtime = []
des_perf_task_runtime_10000 = []
des_perf_task_runtime_20000 = []
des_perf_task_runtime_50000 = []

# Generate some example data
x_range = (0, 7500)

# _filtered_task_runtime_0 = [x for x in task_runtime_0 if x >= x_range[0] and x <= x_range[1]]
# _filtered_task_runtime_2000 = [x for x in task_runtime_2000 if x >= x_range[0] and x <= x_range[1]]

_filtered_des_perf_task_runtime = [x for x in des_perf_task_runtime if x >= x_range[0] and x <= x_range[1]]
_filtered_des_perf_task_runtime_10000 = [x for x in des_perf_task_runtime_10000 if x >= x_range[0] and x <= x_range[1]]
_filtered_des_perf_task_runtime_20000 = [x for x in des_perf_task_runtime_20000 if x >= x_range[0] and x <= x_range[1]]
_filtered_des_perf_task_runtime_50000 = [x for x in des_perf_task_runtime_50000 if x >= x_range[0] and x <= x_range[1]]


# Create a figure and axis
plt.figure(figsize=(10, 6))
plt.title("des_perf(ftask part)")

# Plot histogram for data1
hist1, bins1, _ = plt.hist(_filtered_des_perf_task_runtime, bins=100, alpha=0.5, label="#merge = 0", color='grey')
hist2, bins2, _ = plt.hist(_filtered_des_perf_task_runtime_10000, bins=100, alpha=0.5, label="#merge = 10000", color='red')
hist3, bins3, _ = plt.hist(_filtered_des_perf_task_runtime_20000, bins=100, alpha=0.5, label="#merge = 20000", color='green')
hist4, bins4, _ = plt.hist(_filtered_des_perf_task_runtime_50000, bins=100, alpha=0.5, label="#merge = 50000", color='black')

# Plot histogram for data2
# hist2, bins2, _ = plt.hist(_filtered_bruntime, bins=100, alpha=0.5, label="btask")

# Annotate bars with counts
for count, x, y in zip(hist1, bins1[:-1], hist1):
    plt.text(x, y, str(int(count)), ha='center', va='bottom', color='grey')

for count, x, y in zip(hist2, bins2[:-1], hist2):
    plt.text(x, y, str(int(count)), ha='center', va='bottom', color='red')

for count, x, y in zip(hist3, bins3[:-1], hist3):
    plt.text(x, y, str(int(count)), ha='center', va='bottom', color='green')

for count, x, y in zip(hist4, bins4[:-1], hist4):
    plt.text(x, y, str(int(count)), ha='center', va='bottom', color='black')

# Add labels and legend
plt.xlabel("runtime(ns)")
plt.ylabel("#tasks")
plt.legend()

# Set a specific range for the x-axis
plt.xlim(x_range)  # Replace -3 and 3 with your desired range

# Show the combined histogram
plt.show()




