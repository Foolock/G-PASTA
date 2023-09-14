import matplotlib.pyplot as plt
import numpy as np

def add_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    
    return [x + y for x, y in zip(list1, list2)]

# des_perf btask

# Generate some sample data
x = [0, 10000, 20000, 50000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000] 
y1 = [45.553, 44.112, 44.082, 39.615, 32.500, 29.719, 27.301, 24.825, 22.736, 21.095, 20.185, 19.515, 20.535, 31.224, 40.240] 
y2 = [26.314, 26.449, 26.241, 25.785, 25.638, 25.290, 25.115, 24.820, 24.348, 23.793, 23.139, 21.820, 21.132, 18.650, 13.066]
y3 = add_lists(y1, y2)

# Create a new figure
plt.figure(figsize=(10, 6))

# Plot the data with labels and colors
plt.plot(x, y1, label='taskflow runtime', color='blue', marker='o')
plt.plot(x, y2, label='taskflow buildtime', color='green', marker='s')
plt.plot(x, y3, label='total time', color='red', marker='x')

# Add title and labels
plt.title('des_perf(btasks part)')
plt.xlabel('number of merging')
plt.ylabel('time(ms)')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# des_perf ftask
# Generate some sample data
x = [0, 10000, 20000, 50000] 
y1 = [161.605, 161.910, 167.794, 161.903] 
y2 = [25.688, 25.785, 25.831, 23.584] 

# Create a new figure
plt.figure(figsize=(10, 6))

# Plot the data with labels and colors
line1, = plt.plot(x, y1, 'b-', marker='o', label='taskflow runtime')
line2, = plt.plot(x, y2, 'g-', marker='s', label='taskflow buildtime')

# Add title and labels
plt.title('des_perf(ftasks part)')
plt.xlabel('number of merging')
plt.ylabel('time(ms)')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


