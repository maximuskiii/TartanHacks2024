import cv2
import numpy as np


def id_paths(image_path, npy_array_path, bar_height):
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

    black_img = np.zeros_like(img)
    # Draw contours on the original image
    cv2.drawContours(black_img, contours, -1, (255, 255, 255), 0)

    
    #Cropping
    height, width = black_img.shape[:2]
    new_height = height - bar_height
    cropped_black_img = black_img[0:new_height, 0:width]

    # Show the image with paths highlighted
    cv2.imshow('Paths Identified', cropped_black_img)
    
    while True:
        # If the ESC key is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == 27: # 27 is the ASCII code for the ESC key
            break
            
    cv2.waitKey(0)

    cv2.imwrite(image_path, cropped_black_img)

    # Save the cropped image to a file
    np.save(npy_array_path, cropped_black_img)
    print(f"Image saved to {image_path}")

    cv2.destroyAllWindows()

# Replace 'path_to_your_image.jpg' with the path to the image you want to process
id_paths('Maps+Paths/test.jpg', 'arr.npy', 50)
