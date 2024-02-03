import numpy as np
import subprocess
import matplotlib.pyplot as plt
from matplotlib.image import imread


def calculate_trajectory_and_save(array, output_image_path='graph.png'):
    cpp_executable_path = './coverage_planning'
    array = np.all(array == [0, 0, 0], axis=-1)
    array = np.where(array, 0, 1)
    np.savetxt('input_grid.txt', array, fmt='%d')
    subprocess.run([cpp_executable_path])
    output_path = np.loadtxt('output_path.txt', dtype='int')

    if output_path.ndim == 1:
        output_path = [tuple(output_path)]
    else:
        output_path = [tuple(row) for row in output_path]

    output_path_array = np.array(output_path)

    map_image = array

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(map_image, cmap='Greys', origin='lower', extent=[
              0, map_image.shape[1], 0, map_image.shape[0]])

    x_coords, y_coords = output_path_array[:,
                                           1] + 0.5, output_path_array[:, 0] + 0.5
    ax.plot(x_coords, y_coords, 'r-', linewidth=2, marker='o', markersize=5)
    ax.set_title('Coverage Path on Grid')

    plt.savefig(output_image_path)
    plt.close() 

    return output_path
