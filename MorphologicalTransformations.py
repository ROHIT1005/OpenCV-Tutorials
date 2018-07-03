import cv2
import numpy as np

img = cv2.imread('image.jpg',cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(img, img, mask = mask)

kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(mask, kernel, iterations=1)
dilation = cv2.dilate(mask, kernel, iterations=1)

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# It is the difference between erosion image and dilation of the image
gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('Gradient',gradient)

# It is the difference between input image and Opening of the image
tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('Tophat',tophat)

# It is the difference between the closing of the input image and input image.
blackhat = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('Blackhat',blackhat)


cv2.imshow('opening',opening)
cv2.imshow('closing',closing)
cv2.imshow('Dilation',dilation)
cv2.imshow('Erosion',erosion)
cv2.imshow('img',img)
cv2.imshow('result',result)

cv2.waitKey(0)
cv2.destroyAllWindows()
