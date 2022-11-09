import cv2 as cv
import math
import time
import os

imgname = "fotografietest.png"

img = cv.imread(imgname, cv.IMREAD_COLOR)
rows, cols, _ = img.shape
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, invbinaryimg = cv.threshold(grayimg, 230, 255, cv.THRESH_BINARY_INV)

contrastarr = [[0 for j in range(cols)] for i in range(rows)]

for i in range(rows):
    for j in range(cols):
        if (i > 0):
            contrastarr = max(contrastarr, abs(grayimg[i][j] - gray[i-1][j]))

cv.imshow("grayimg", grayimg)
cv.imshow("invbinaryimg", invbinaryimg)
    
cv.waitKey(0)
cv.destroyAllWindows()