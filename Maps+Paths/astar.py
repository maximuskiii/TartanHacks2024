import cv2 
import numpy as np

image = np.loadtxt('your_file_path.txt')

cv2.imshow("test", image)

cv2.waitKey(0)

cv2.imwrite("new_test.jpg", image)

cv2.destroyAllWindows()

print(image)