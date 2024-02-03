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

    black_img = np.zeros_like(img)
    # Draw contours on the original image
    cv2.drawContours(black_img, contours, -1, (255, 255, 255), 2)

    # Show the image with paths highlighted
    cv2.imshow('Paths Identified', black_img)
    
    
    while True:
        # If the ESC key is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == 27: # 27 is the ASCII code for the ESC key
            break
            
    cv2.waitKey(0)

    cv2.imwrite('test.jpg', black_img)
    cv2.destroyAllWindows()

# Replace 'path_to_your_image.jpg' with the path to the image you want to process
identify_paths('test.jpg')


def crop_and_save_image(image_path, bar_height, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image could not be read.")
        return
    
    # Get the height and width of the image
    height, width = img.shape[:2]
    
    # Calculate the new height by subtracting the bar height
    new_height = height - bar_height
    
    # Crop the image to remove the bar from the bottom
    # Note: img[y1:y2, x1:x2] where (x1,y1) is the top-left coordinate
    # and (x2,y2) is the bottom-right coordinate of the cropped area
    cropped_img = img[0:new_height, 0:width]
    
    # Save the cropped image to a file
    cv2.imwrite(output_path, cropped_img)
    np.save('arr.npy', cropped_img)
    print(f"Image saved to {output_path}")

# Replace 'path_to_your_image.jpg' with the path to the image you want to process
# Set 'bar_height' to the height of the bar you want to remove from the bottom
# Replace 'output_path.jpg' with the path where you want to save the cropped image
crop_and_save_image('test.jpg', 50, 'test.jpg')

