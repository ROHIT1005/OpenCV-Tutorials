import cv2
import numpy as np

img = cv2.imread('image2.jpg',cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

#laplacian2 = cv2.Laplacian(img2, cv2.CV_64F)
#sobelx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)
#sobely2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)

#cv2.imshow('SobelX2',sobelx2)
#cv2.imshow('SobelY2',sobely2)
#cv2.imshow('original2',img2)
#cv2.imshow('Laplacian2',laplacian2)

edges = cv2.Canny(img, 100, 200)

cv2.imshow('Edges',edges)
cv2.imshow('SobelX',sobelx)
cv2.imshow('SobelY',sobely)
cv2.imshow('original',img)
cv2.imshow('Laplacian',laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
