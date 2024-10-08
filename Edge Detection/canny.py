import cv2
import numpy as np


img = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Sobel Operator

sobelx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_8U, 0, 1, ksize=3)

# Combine the edges in x and y directions

edges = cv2.bitwise_or(sobelx, sobely)

# Canny Edge

canny = cv2.Canny(edges, 100, 200)

cv2.imshow('Original Image', img)
cv2.imwrite('edges.jpg', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
