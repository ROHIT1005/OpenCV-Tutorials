import cv2
import numpy as np

# 500 x 250
img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

add = img1+img2
add1 = cv2.add(img1,img2)
weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
#img1,weight,img2,weight,gamma_value

cv2.imshow('weighted',weighted)
cv2.imshow('add',add)
cv2.imshow('add1',add1)
cv2.waitKey(0)
cv2.destroyAllWindows()
