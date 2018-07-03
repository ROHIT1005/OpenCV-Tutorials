import cv2
import numpy as np

img = cv2.imread('image.jpg',cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(img, img, mask = mask)


#Operation is like this: keep this kernel above a pixel, add all the 25 pixels below this kernel, take its average and replace the central pixel with the new average value. It continues this operation for all the pixels in the image. 

kernel = np.ones((5,5),np.float32)/25
smoothed = cv2.filter2D(result,-1,kernel)
    
#In this, instead of box filter, gaussian kernel is used. It is done with the function, cv2.GaussianBlur(). We should specify the width and height of kernel which should be positive and odd. We also should specify the standard deviation in X and Y direction, sigmaX and sigmaY respectively. If only sigmaX is specified, sigmaY is taken as same as sigmaX. If both are given as zeros, they are calculated from kernel size. Gaussian blurring is highly effective in removing gaussian noise from the image.

gausblur = cv2.GaussianBlur(result,(15,15),0)

#Here, the function cv2.medianBlur() takes median of all the pixels under kernel area and central element is replaced with this median value. This is highly effective against salt-and-pepper noise in the images. Interesting thing is that, in the above filters, central element is a newly calculated value which may be a pixel value in the image or a new value. But in median blurring, central element is always replaced by some pixel value in the image. It reduces the noise effectively. Its kernel size should be a positive odd integer.

median = cv2.medianBlur(result,15)

#Bilateral filter also takes a gaussian filter in space, but one more gaussian filter which is a function of pixel difference. Gaussian function of space make sure only nearby pixels are considered for blurring while gaussian function of intensity difference make sure only those pixels with similar intensity to central pixel is considered for blurring. So it preserves the edges since pixels at edges will have large intensity variation.

bilateral = cv2.bilateralFilter(result,15,75,75)

cv2.imshow('bilateral Blur',bilateral)
cv2.imshow('Median Blur',median)
cv2.imshow('Gaussian Blurring',gausblur)
cv2.imshow('Averaging',smoothed)
cv2.imshow('img',img)
cv2.imshow('result',result)
cv2.imshow('mask',mask)
cv2.imshow('hsv',hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
