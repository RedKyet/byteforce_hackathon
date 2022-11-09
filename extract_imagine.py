'''
import cv2
# read image without color
image = cv2.imread('Assets\\Greyscale test\\start.png')
"""cv2.imshow('Greyscaled',img)
cv2.waitKey(0)"""

original = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
#dilate = cv2.dilate(thresh, kernel, iterations=1)

# Find contours, obtain bounding box coordinates, and extract ROI
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]
    cv2.imwrite("ROI_{}.png".format(image_number), ROI)
    image_number += 1

cv2.imshow('image', image)
cv2.imshow('gray', gray)
cv2.imshow('thresh', thresh)
#cv2.imshow('dilate', dilate)
cv2.waitKey()

#cv2.imwrite('Assets\\Greyscale test\\rock_greyscaled.jpg',img)

'''

import cv2 as cv
import math
import time
import os
import numpy as np

##############################################################################################################

imgname = "fotografietest.png"
binarythresh = 240
contrastthresh = 10

##############################################################################################################


img = cv.imread(imgname, cv.IMREAD_COLOR)
rows, cols, _ = img.shape
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

contrastarr = [[0 for j in range(cols)] for i in range(rows)]

for i in range(rows):
    for j in range(cols):
        if (i > 0):
            contrastarr[i][j] = max(contrastarr[i][j], np.uint8(abs(int(grayimg[i][j]) - int(grayimg[i-1][j]))))
        if (i < rows-1):
            contrastarr[i][j] = max(contrastarr[i][j], np.uint8(abs(int(grayimg[i][j]) - int(grayimg[i+1][j]))))
        if (j > 0):
            contrastarr[i][j] = max(contrastarr[i][j], np.uint8(abs(int(grayimg[i][j]) - int(grayimg[i][j-1]))))
        if (j < cols-1):
            contrastarr[i][j] = max(contrastarr[i][j], np.uint8(abs(int(grayimg[i][j]) - int(grayimg[i][j+1]))))

invbinaryimg = cv.threshold(grayimg, binarythresh, 255, cv.THRESH_BINARY_INV)[1]

for i in range(rows):
    for j in range(cols):
        if invbinaryimg[i][j] == 0 and contrastarr[i][j] < contrastthresh:
            #invbinaryimg[i][j] = 255
            pass

cv.imshow("grayimg", grayimg)
cv.imshow("invbinaryimg", invbinaryimg)

cv.waitKey(0)
cv.destroyAllWindows()