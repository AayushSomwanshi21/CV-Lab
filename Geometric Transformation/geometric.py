import cv2
import numpy as np

img = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck.jpg')
angle = 45

# h = rows , w = cols

h, w = img.shape[:2]

# Rotation
M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
rotated_img = cv2.warpAffine(img, M, (w, h))
cv2.imshow('Rotated Image', rotated_img)

# Scaling

img_shrink = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
cv2.imshow('Shrunked Image', img_shrink)
img_enlarged = cv2.resize(img_shrink, None, fx=1.5,
                          fy=1.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Enlarged Image', img_enlarged)

# Translation

M_translation = np.float32([[1, 0, 100], [0, 1, 50]])

img_translated = cv2.warpAffine(img, M_translation, (w, h))
cv2.imshow('Translated Image', img_translated)

# Shear

M_shear = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])

img_shear = cv2.warpPerspective(img, M_shear, (int(w*1.5), int(h*1.5)))
cv2.imshow('Sheared Image', img_shear)

cv2.waitKey(0)
cv2.destroyAllWindows()
