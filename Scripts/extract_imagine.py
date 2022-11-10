import cv2 as cv
import math
import time
import os
import numpy as np

##############################################################################################################

imgname = "Assets\\Greyscale test\\fotografietest.png"
binarythresh = 240
contrastthresh = 20

##############################################################################################################


img = cv.imread(imgname, cv.IMREAD_COLOR)
rows, cols, _ = img.shape

# algoritmul lui lee pt gasirea fundalului

contrastarr = [[0 for j in range(cols)] for i in range(rows)]

cnt = 0
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]
nrfilled = 0
bbxmini = 999999
bbxmaxi = -1
bbymini = 999999
bbymaxi = -1

bgbb = 0
bgbbarea = -1

queue = []


for i in range(rows):
    for j in range(cols):
        if contrastarr[i][j] == 0:
            cnt += 1
            nrfilled = 0
            bbxmini = 999999
            bbxmaxi = -1
            bbymini = 999999
            bbymaxi = -1
            queue.append((i, j))
            contrastarr[i][j] = cnt
            while len(queue) > 0:
                i = queue[0][0]
                j = queue[0][1]
                queue.pop()
                if i < bbxmini:
                    bbxmini = i
                if i > bbxmaxi:
                    bbxmaxi = i
                if j < bbymini:
                    bbymini = j
                if j > bbymaxi:
                    bbymaxi = j
                nrfilled += 1
                color1 = img[i][j]
                color1 = [int(el) for el in color1]
                for ind in range(4):
                    x = i+dx[ind]
                    y = j+dy[ind]
                    if (x >= 0 and x < rows and y >= 0 and y < cols):
                        if contrastarr[x][y] == 0:
                            color2 = img[x][y]
                            color2 = [int(el) for el in color2]
                            mandist = abs(color1[0]-color2[0]) + abs(color1[1]-color2[1]) + abs(color1[2]-color2[2])
                            if mandist < contrastthresh:
                                contrastarr[x][y] = cnt
                                queue.append((x, y))
            if (bbxmaxi-bbxmini+1)*(bbymaxi-bbymini+1) > bgbbarea:
                bgbbarea = (bbxmaxi-bbxmini+1)*(bbymaxi-bbymini+1)
                bgbb = cnt

print(cnt)
print([bbxmini, bbymini, bbxmaxi, bbymaxi])

wbimg = img.copy()
for i in range(rows):
    for j in range(cols):
        if contrastarr[i][j] == bgbb:
            wbimg[i][j] = (np.uint8(255), np.uint8(255), np.uint8(255))

cv.imshow("wbimg", wbimg)
cv.waitKey(0)
cv.destroyAllWindows()

quit()


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

# Show the output image
cv.imshow('Output', out)
cv.waitKey(0)
cv.destroyAllWindows()
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