import cv2
import numpy as np

img = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

C = 20

log_img = C*np.log(1 + gray)
# log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
log_img = np.uint8(log_img)
result = cv2.hconcat([gray, log_img])
cv2.imshow('Log Tranformation', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
