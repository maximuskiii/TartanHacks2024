import cv2
import numpy as np


def identify_paths(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image could not be read.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # Show the image with paths highlighted
    cv2.imshow('Paths Identified', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'path_to_your_image.jpg' with the path to the image you want to process
identify_paths('path_to_your_image.jpg')