import cv2
import numpy as np

img = cv2.imread('image.jpg',cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow('img',img)
cv2.imshow('result',result)
cv2.imshow('mask',mask)
cv2.imshow('hsv',hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
