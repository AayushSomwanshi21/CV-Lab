import cv2
import numpy as np

img1 = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck.jpg')
img2 = cv2.imread('C:/Users/Lenovo/Documents/VSCODE/CV Lab/duck_rotated.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create a SIFT object

sift = cv2.SIFT_create()

# find keypoints and descriptors

key1, desc1 = sift.detectAndCompute(gray1, None)
key2, desc2 = sift.detectAndCompute(gray2, None)

# Brute Force(BF) Matcher

bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
matches = bf.match(desc1, desc2)

matches = sorted(matches, key=lambda x: x.distance)

sift_image = cv2.drawMatches(
    img1, key1, img2, key2, matches[:50], img2, flags=2)

cv2.imshow('SIFT Image', sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
