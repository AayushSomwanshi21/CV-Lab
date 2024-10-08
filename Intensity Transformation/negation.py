import cv2

img = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
neg = cv2.bitwise_not(gray)
cv2.imshow('Input Image', img)
cv2.imshow('Negative Image', neg)
cv2.waitKey(0)
cv2.destroyAllWindows()
