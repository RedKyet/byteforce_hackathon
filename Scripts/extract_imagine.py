import cv2 as cv
import math
import time
import os
import numpy as np

##############################################################################################################

imgname = "Assets\\Greyscale test\\fotografietest.png"
binarythresh = 240
contrastthresh = 40

##############################################################################################################


img = cv.imread(imgname, cv.IMREAD_COLOR)
rows, cols, _ = img.shape
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
invbinaryimg = cv.threshold(grayimg, binarythresh, 255,cv.THRESH_BINARY)[1]

contours = cv.findContours(invbinaryimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
contoursbinaryimg = invbinaryimg.copy()
cv.drawContours(contoursbinaryimg, contours, -1, color=(255, 255, 255), thickness=cv.FILLED)

contrastarr = [[0 for j in range(cols)] for i in range(rows)]
'''
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
'''
betterholesimg = invbinaryimg.copy()

for i in range(rows):
    for j in range(cols):
        if invbinaryimg[i][j] == 255 and contoursbinaryimg[i][j] == 255:
            cnt = 0
            if i > 0 and invbinaryimg[i-1][j] == 0:
                cnt += 1
            if i < rows-1 and invbinaryimg[i+1][j] == 0:
                cnt += 1
            if j > 0 and invbinaryimg[i][j-1] == 0:
                cnt += 1
            if j < cols-1 and invbinaryimg[i][j+1] == 0:
                cnt += 1
            if cnt > 0:
                betterholesimg[i][j] = 0

#save images
cnts = cv.findContours(betterholesimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(type(cnts))
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number = 0

###############################################################################
#save min and max area image
minar_img=min(cnts,key=lambda x:cv.contourArea(x))
maxar_img=max(cnts,key=lambda x:cv.contourArea(x))
x,y,w,h = cv.boundingRect(minar_img)
element = img[y:y+h, x:x+w]
cv.imwrite("Assets\\Objects\\element_min_area_{}.png".format(cv.contourArea(minar_img)), element)
x,y,w,h = cv.boundingRect(maxar_img)
element = img[y:y+h, x:x+w]
cv.imwrite("Assets\\Objects\\element_max_area_{}.png".format(cv.contourArea(maxar_img)), element)
#
#
#save min and max perim image
minper_img=min(cnts,key=lambda x:cv.arcLength(x, True))
maxper_img=max(cnts,key=lambda x:cv.arcLength(x, True))
x,y,w,h = cv.boundingRect(minper_img)
element = img[y:y+h, x:x+w]
cv.imwrite("Assets\\Objects\\element_min_perim_{}.png".format(cv.arcLength(minper_img, True)), element)
x,y,w,h = cv.boundingRect(maxper_img)
element = img[y:y+h, x:x+w]
cv.imwrite("Assets\\Objects\\element_max_perim_{}.png".format(cv.arcLength(maxper_img, True)), element)
################################################################################
#save min and max brightness image
"""intensities = []
for i in range(len(cnts)):
    cimg = np.zeros_like(img)
    cv.drawContours(cimg, [contours[i]], -1, color=(255,255,255), thickness=-1)
    pts = np.where(cimg == 255)
    intensities.append(img[pts[0], pts[1]])
print(intensities[0])

maxi = 0
intensarray = [0] * len(cnts)
for i in range(len(cnts)):
    intensarray[i] = i

for i in range(len(cnts)-1):
    for j in range(i+1, len(cnts)):
        if sum(sum(intensities[i])/len(intensities[i])) > sum(sum(intensities[j])/len(intensities[j])):
            aux = intensities[i]
            intensities[i] = intensities[j]
            intensities[j] = aux
            aux2 = intensarray[i]
            intensarray[i] = intensarray[j]
            intensarray[j] = aux2

i = intensarray[0]
j = intensarray[len(cnts)-1]

cv.imshow("img1", img)

print(sum(sum(intensities[i]))/len(intensities[i]))
print(sum(sum(intensities[j]))/len(intensities[j]))
cv.drawContours(img, [contours[i]], -1, color=(255,0,0), thickness=-1)
cv.drawContours(img, [contours[j]], -1, color=(0,255,0), thickness=-1)"""

################################################################################
#fill contour
cv.drawContours(img,cnts,0,thickness=cv.FILLED,color=[255,255,255])
cv.drawContours(img,cnts,0,thickness=5,color=[int(s) for s in img[0][0]])
################################################################################

for c in cnts:
    x,y,w,h = cv.boundingRect(c)
    #cv.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    
    #draw contour
    #cv.drawContours(img, [c], -1, (0,255,255), 1)
    
    element = img[y:y+h, x:x+w] #selectam imaginea intr-un dreptunghi

    cv.imwrite("Assets\\Objects\\element_{}_area_{}_perim_{}.png".format(image_number,cv.contourArea(c),round(cv.arcLength(c, True),4)), element)
    image_number += 1

cv.imshow("img", img)
cv.imshow("grayimg", grayimg)
cv.imshow("invbinaryimg", invbinaryimg)
cv.imshow("betterholesimg", betterholesimg)

cv.waitKey(0)
cv.destroyAllWNHMindows()

'''
COLT DE DAT CU MNJMNJMJNNHMJ IN TASTATURA



HYJUNHJMNHJHJNMNHJMNHJNHJMNHJNMNHJMJ.,,M,M

'''