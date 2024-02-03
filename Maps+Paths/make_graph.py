import cv2
import numpy as np


def id_paths(image_path, npy_array_path, bar_height):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: img can't be read")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_img = np.zeros_like(img)
    cv2.drawContours(black_img, contours, -1, (255, 255, 255), 4)

    
    height, width = black_img.shape[:2]
    new_height = height - bar_height
    cropped_black_img = black_img[0:new_height, 0:width]

    cv2.imshow('Paths Identified', cropped_black_img)
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cv2.waitKey(0)

    cv2.imwrite(image_path, cropped_black_img)
    np.save(npy_array_path, cropped_black_img)
    print(f"Image saved to {image_path}")

    cv2.destroyAllWindows()

id_paths('/Users/myagnyatinskiy/Desktop/TartanHacks2024/Maps+Paths/test.jpg', '/Users/myagnyatinskiy/Desktop/TartanHacks2024/Maps+Paths/arr.npy', 0)
