import cv2 as cv
import math
import time
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw


from adaugare_numar import adaugare_numar
from concatenare import concatenare_orizontala
from detect_culoare import get_color
from obtinere_index import obtinere_index

##############################################################################################################

imgname = "Assets\\Greyscale test\\fotografietestalbastra.png"
binarythresh = 240
contrastthresh = 20
epsilonarie = 5.0
epsilonper = 1.0

##############################################################################################################


img = cv.imread(imgname, cv.IMREAD_COLOR)
rows, cols, _ = img.shape

# turn background to white

bgbb = (0, 0, 0, 0)
bgbbarea = -1
bgcolor = (0, 0, 0)

wbimg = img.copy()
rect = cv.floodFill(wbimg, None, (0, 0), (255, 255, 255), loDiff=5, upDiff=5)[3]
rectarea = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)
if rectarea > bgbbarea:
    bgbb = rect
    bgbbarea = rectarea
    bgcolor = img[0][0]
    finalwbimg = wbimg.copy()

wbimg = img.copy()
rect = cv.floodFill(wbimg, None, (0, rows-1), (255, 255, 255), loDiff=5, upDiff=5)[3]
rectarea = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)
if rectarea > bgbbarea:
    bgbb = rect
    bgbbarea = rectarea
    bgcolor = img[rows-1][0]
    finalwbimg = wbimg.copy()

wbimg = img.copy()
rect = cv.floodFill(wbimg, None, (cols-1, 0), (255, 255, 255), loDiff=5, upDiff=5)[3]
rectarea = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)
if rectarea > bgbbarea:
    bgbb = rect
    bgbbarea = rectarea
    bgcolor = img[0][cols-1]
    finalwbimg = wbimg.copy()

wbimg = img.copy()
rect = cv.floodFill(wbimg, None, (cols-1, rows-1), (255, 255, 255), loDiff=5, upDiff=5)[3]
rectarea = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)
if rectarea > bgbbarea:
    bgbb = rect
    bgbbarea = rectarea
    bgcolor = img[rows-1][cols-1]
    finalwbimg = wbimg.copy()

bgcolor = [int(s) for s in bgcolor]

wbimg = finalwbimg.copy()

# get object contours

grayimg = cv.cvtColor(wbimg, cv.COLOR_BGR2GRAY)
invbinaryimg = cv.threshold(grayimg, binarythresh, 255,cv.THRESH_BINARY_INV)[1]

contours = cv.findContours(invbinaryimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
contoursbinaryimg = invbinaryimg.copy()
cv.drawContours(contoursbinaryimg, contours, -1, color=(255, 255, 255), thickness=cv.FILLED)

# give indexes to objects

indexedimg = np.zeros((rows, cols, 3), np.uint8)

for i in range(len(contours)):
    cr = ((i+1) & 255)
    cg = (((i+1) >> 8) & 255)
    cb = (((i+1) >> 16) & 255)
    cv.drawContours(indexedimg, [contours[i]], -1, color=(cb, cg, cr), thickness=cv.FILLED)

# make inner holes more accurate (optional)
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

invbinaryimg = betterholesimg.copy()
'''
cv.waitKey(0)

#save min and max area image

objectareas = [0] * len(contours)
objectperimeters = [0] * len(contours)

maxarie = -1
for i in range(len(contours)):
    if cv.contourArea(contours[i]) > maxarie:
        maxarie = cv.contourArea(contours[i])
    objectareas[i] = cv.contourArea(contours[i])

minper = 9999999
for i in range(len(contours)):
    if cv.arcLength(contours[i], True) < minper:
        minper = cv.arcLength(contours[i], True)
    objectperimeters[i] = cv.arcLength(contours[i], True)

onlymaxarieminper = img.copy()

for i in range(len(contours)):
    currarie = cv.contourArea(contours[i])
    currper = cv.arcLength(contours[i], True)
    if maxarie-currarie > epsilonarie and currper-minper > epsilonper:
        cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=cv.FILLED)

cv.imshow("only max arie and min per", onlymaxarieminper)
cv.imshow("invbinaryimg", invbinaryimg)
cv.waitKey(0)

#save min and max brightness image
intensities = [0] * len(contours)
nrpixels = [0] * len(contours)

for i in range(rows):
    for j in range(cols):
        aux = obtinere_index(indexedimg, i, j)
        if aux >= 0:
            intensities[aux] += (int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2]))
            nrpixels[aux] += 1



for i in range(len(contours)-1):
    for j in range(i+1, len(contours)):
        if sum(sum(intensities[i])/len(intensities[i])) > sum(sum(intensities[j])/len(intensities[j])):
            aux = intensities[i]
            intensities[i] = intensities[j]
            intensities[j] = aux
            aux2 = intensarray[i]
            intensarray[i] = intensarray[j]
            intensarray[j] = aux2

i = intensarray[0]
j = intensarray[len(contours)-1]

cv.imshow("img1", img)

print(sum(sum(intensities[i]))/len(intensities[i]))
print(sum(sum(intensities[j]))/len(intensities[j]))
cv.drawContours(img, [contours[i]], -1, color=(255,0,0), thickness=-1)
cv.drawContours(img, [contours[j]], -1, color=(0,255,0), thickness=-1)

blanc = np.zeros((1, 1, 3), np.uint8)

aux = Image.open('Assets\\Greyscale test\\Gaina4.jpg')
cul = get_color(aux)

cv.imwrite("Assets\\Objects\\rez.png".format(image_number), blanc)

################################################################################
for c in contours:
    x,y,w,h = cv.boundingRect(c)
    #cv.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    
    #draw contour
    #cv.drawContours(img, [c], -1, (0,255,255), 1)
    
    element = img[y:y+h, x:x+w] #selectam imaginea intr-un dreptunghi

    cv.imwrite("Assets\\Objects\\element_{}_area_{}_perim_{}.png".format(image_number,cv.contourArea(c),round(cv.arcLength(c, True),4)), element)


    #creere imagnie cu numar
    im = Image.open("Assets\\Objects\\element_{}_area_{}_perim_{}.png".format(image_number,cv.contourArea(c),round(cv.arcLength(c, True),4)))
    adaugare_numar(im, image_number)
    image_number += 1


    
    rez = Image.open("Assets\\Objects\\rez.png")
    
    #originalImage = cv.imread(concatenare_orizontala(rez, im, (0, 0, 0)))
    #cv.imwrite("Assets\\Objects\\rez.png", concatenare_orizontala(rez, im, (0, 0, 0)))
    concatenare_orizontala(rez, im, cul).save("Assets\\Objects\\rez.png")
   # rez.save("Assets\\Objects\\rez.png")
    #creere imagine prin concatenare
    
    
cv.imshow("img", img)

cv.waitKey(0)
cv.destroyAllWNHMindows()

'''
COLT DE DAT CU MNJMNJMJNNHMJ IN TASTATURA



HYJUNHJMNHJHJNMNHJMNHJNHJMNHJNMNHJMJ.,,M,M

'''