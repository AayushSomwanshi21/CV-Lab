import cv2
import numpy as np


img = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck.jpg')

# dividing by 25 since there are 5X5 elements in the kernel and we are calc the mean filter
kernel = np.ones((5, 5), np.float32) / 25

mean_filtered = cv2.filter2D(img, -1, kernel)
median_filtered = cv2.medianBlur(img, 5)
laplacian_filtered = cv2.Laplacian(img, cv2.CV_64F)

cv2.imshow('Original Image', img)
cv2.imshow('Mean Filtered Image', mean_filtered)
cv2.imshow('Median Filtered Image', median_filtered)
cv2.imshow('Laplacian Filtered Image', laplacian_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
