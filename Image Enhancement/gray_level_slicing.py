import cv2
import numpy as np

# 2nd argument (0) to read the img in gray scale
img = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck.jpg', 0)

row, col = img.shape
# create a zeroes array
img1 = np.zeros((row, col), dtype='uint8')

min_range = 40
max_range = 140

for i in range(row):
    for j in range(col):

        if img[i, j] > min_range and img[i, j] < max_range:
            img1[i, j] = 255
        else:
            if i > 0 and j > 0:  # To handle the edge condition
                img1[i, j] = img[i-1, j-1]
            else:
                img1[i, j] = 0

cv2.imwrite('EnhancedImage.jpg', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
