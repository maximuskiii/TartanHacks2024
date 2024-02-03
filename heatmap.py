import numpy as np
import cv2


def load_npy(file_path):
    return np.load(file_path)


if __name__ == "__main__":
    # Load the heatmap
    heatmap = load_npy("heatmap.npy")

    # Define circle parameters
    center_x, center_y = heatmap.shape[0] // 2 + 13, heatmap.shape[1] // 2
    radius = (
        min(center_x, center_y) // 2 - 10
    )  # Adjust this factor to change the size of the circle

    # Apply gradient based on circle
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if distance <= radius and heatmap[i][j][0] > 1:
                intensity = int(255 * (distance / radius))
                heatmap[i][j] = [intensity, intensity, intensity]

    # invert the heatmap
    # heatmap = cv2.bitwise_not(heatmap)

    # apply color map
    heatmap = cv2.applyColorMap(heatmap[:, :, 0], cv2.COLORMAP_JET)

    # Display the heatmap
    cv2.imshow("Heatmap", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
