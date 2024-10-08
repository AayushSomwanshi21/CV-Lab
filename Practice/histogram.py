import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20, 20))

plt.subplot(221)
plt.title('Original Image')
plt.imshow(img)

plt.subplot(222)
plt.title('Histogram')
hist_img = cv2.calcHist(img1, [0], None, [256], [0, 256])
plt.plot(hist_img)

plt.subplot(223)
plt.title('Histogram 1')
plt.hist(img1.ravel(), 256, [0, 256])

final_img = cv2.equalizeHist(img1)

plt.subplot(224)
plt.title('Equalized Image')
plt.imshow(final_img)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
