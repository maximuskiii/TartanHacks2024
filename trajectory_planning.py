import numpy as np
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Pre-existing setup (unchanged)

cpp_executable_path = './coverage_planning'
array = np.load('arr-2.npy')
array = np.all(array == [0, 0, 0], axis=-1)
array = np.where(array, 0, 1)
np.savetxt('input_grid.txt', array, fmt='%d')
subprocess.run([cpp_executable_path])
output_path = np.loadtxt('output_path.txt', dtype='int')
if output_path.ndim == 1:
    output_path = [tuple(output_path)]
else:
    output_path = [tuple(row) for row in output_path]

# Convert output_path to a NumPy array for easier handling
output_path_array = np.array(output_path)

# Setup figure for subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# Plot for the first subplot (grid with path placeholder)
axes[0].imshow(array, cmap='Greys', origin='lower', extent=[0, array.shape[1], 0, array.shape[0]])
line, = axes[0].plot([], [], 'r-', linewidth=2, marker='o', markersize=5)  # Line object for animating the path

axes[0].set_title('Coverage Path on Grid')

# Static plot for the second subplot (grid without path)
axes[1].imshow(array, cmap='Greys', origin='lower', extent=[0, array.shape[1], 0, array.shape[0]])
axes[1].set_title('Grid Without Path')

# Initialize animation function
def init():
    line.set_data([], [])
    return line,

# Corrected animation update function
def update(frame):
    x_coords, y_coords = output_path_array[:frame + 1, 1] + 0.5, output_path_array[:frame + 1, 0] + 0.5
    line.set_data(x_coords, y_coords)
    return line,

# Creating the animation
ani = FuncAnimation(fig, update, frames=len(output_path), init_func=init, blit=True, interval=100)

plt.tight_layout()
plt.show()

print("Output path from C++ program:")
print(output_path)
