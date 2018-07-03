import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('opencv-feature-matching-template.jpg',0)
img2 = cv2.imread('opencv-feature-matching-image.jpg',0)

orb = cv2.ORB_create()

#Keypoints and Descriptors
#Detects keypoints and computes the descriptors 
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

#Find Keypoints and Descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

#Use the BF matcher to match the features
matches = bf.match(des1, des2)
#Sort em in ascending order
matches = sorted(matches, key = lambda x:x.distance)

#Draw the matched items upto 10 items
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
plt.imshow(img3)
plt.show()
