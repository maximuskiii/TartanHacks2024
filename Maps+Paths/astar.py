import cv2 
import numpy as np

image = np.loadtxt('/Users/myagnyatinskiy/Desktop/TartanHacks2024/Maps+Paths/matrix_reduced_precision.txt')

cv2.imshow("test", image)

cv2.waitKey(0)

cv2.imwrite("new_test.jpg", image)

cv2.destroyAllWindows()

print(image)