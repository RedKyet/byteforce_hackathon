import cv2 as cv
import math
import time
import os

imgname = input("Enter image name: ")

img = cv.imread(imgname, cv.IMREAD_GRAYSCALE)
rows, cols, _ = img.shape

cv.imshow(img)
cv.waitKey(0)
cv.destroyAllWindows()