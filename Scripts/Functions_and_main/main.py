import cv2 as cv
import math
import time
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
def magic(url):
    dict = {
        
    }
    return dict
"""from adaugare_numar import adaugare_numar
from concatenare import concatenare_orizontala
from detect_culoare import get_color"""
# turn background to white

bgbb = (0, 0, 0, 0)
bgbbarea = -1
bgcolor = (0, 0, 0)
#epsilons
binarythresh = 245
epsilonarie = 5.0
epsilonper = 1
epsilonbright = 0.005
epsilondark = 0.005

#####Functions######
imgname = "Assets\\Greyscale test\\start.png"
img=cv.imread(imgname, cv.IMREAD_COLOR)
rows, cols, _ = img.shape

def get_img():
    global img
    return img

def set_img_path(path: str = "Assets\\Greyscale test\\start.png"):
    global img
    imgname=path
    img = cv.imread(imgname, cv.IMREAD_COLOR)

def obtinere_index(indeximg, line, column):
    return (((int(indeximg[line][column][2])) + ((int(indeximg[line][column][1])) << 8) + ((int(indeximg[line][column][0])) << 16)) - 1)


def adaugare_numar (a, num):

    draw = ImageDraw.Draw(a)

    font = ImageFont.truetype('Scripts\\fontu.ttf', 25)

    text= str(num+1)

    draw.text((5, 5), text= text, fill="red", font=font, align="right")

    a.save("Assets\\Objects\\element_numar_{}.png".format(num))

def concatenare_orizontala(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width,
                    max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

##################


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
invbinaryimg = cv.threshold(grayimg, binarythresh, 255, cv.THRESH_BINARY_INV)[1]

contours = cv.findContours(invbinaryimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
contoursbinaryimg = invbinaryimg.copy()
cv.drawContours(contoursbinaryimg, contours, -1, color=(255, 255, 255), thickness=cv.FILLED)

# sort contours by area in decreasing order

contours = list(contours)
contours.sort(key=lambda c:cv.contourArea(c), reverse=True)

# give indexes to objects

indexedimg = np.zeros((rows, cols, 3), np.uint8)

for i in range(len(contours)):
    cr = ((i+1) & 255)
    cg = (((i+1) >> 8) & 255)
    cb = (((i+1) >> 16) & 255)
    cv.drawContours(indexedimg, [contours[i]], -1, color=(cb, cg, cr), thickness=cv.FILLED)

# define arrays with properties

objectareas = [0] * len(contours)
objectperimeters = [0] * len(contours)
intensities = [0] * len(contours)
nrpixels = [0] * len(contours)

for i in range(rows):
    for j in range(cols):
        ind = obtinere_index(indexedimg, i, j)
        if ind >= 0:
            nrpixels[ind] += 1
            intensities[ind] += (int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2]))


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

ok = False
for i in range(len(contours)):
    currarie = cv.contourArea(contours[i])
    currper = cv.arcLength(contours[i], True)
    if maxarie-currarie <= epsilonarie and currper-minper <= epsilonper:
        ok = True

if not ok:
    for i in range(len(contours)):
        currarie = cv.contourArea(contours[i])
        currper = cv.arcLength(contours[i], True)
        if maxarie-currarie > epsilonarie and currper-minper > epsilonper:
            cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=cv.FILLED)
            cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=8)
else:
    for i in range(len(contours)):
        currarie = cv.contourArea(contours[i])
        currper = cv.arcLength(contours[i], True)
        if maxarie-currarie > epsilonarie or currper-minper > epsilonper:
            cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=cv.FILLED)
            cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=8)

# object brightnesses

maxbrightness = -1.0
minbrightness = 10250.0
for i in range(len(contours)):
    bright = intensities[i] / nrpixels[i]
    if bright > maxbrightness:
        maxbrightness = bright
    if bright < minbrightness:
        minbrightness = bright

onlybrightanddark = img.copy()

for i in range(len(contours)):
    bright = intensities[i] / nrpixels[i]
    if bright-minbrightness > epsilondark and maxbrightness-bright > epsilonbright:
        cv.drawContours(onlybrightanddark, [contours[i]], -1, color=bgcolor, thickness=cv.FILLED)
        cv.drawContours(onlybrightanddark, [contours[i]], -1, color=bgcolor, thickness=8)

# make bg transparent

alphaimg = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
for i in range(rows):
    for j in range(cols):
        ind = obtinere_index(indexedimg, i, j)
        if ind >= 0:
            alphaimg[i][j][3] = 255
        else:
            alphaimg[i][j][3] = 0

# sort objects in decreasing order of areas

image_number=0
blanc = np.zeros((1, 1, 3), np.uint8)
cv.imwrite("Assets\\Objects\\rez.png",blanc)

for c in range(len(contours)):
    x, y, w, h = cv.boundingRect(contours[c])
    element = alphaimg[y:y+h, x:x+w]

    for i in range(h):
        for j in range(w):
            ind = obtinere_index(indexedimg, y+i, x+j)
            if ind is not c:
                element[i][j][3] = 0
                element[i][j][0] = bgcolor[0]
                element[i][j][1] = bgcolor[1]
                element[i][j][2] = bgcolor[2]

    cv.imwrite("Assets\\Objects\\element_{}.png".format(image_number), element)
    im = Image.open("Assets\\Objects\\element_{}.png".format(image_number))

    adaugare_numar(im, image_number)
    image_number += 1
    rez = Image.open("Assets\\Objects\\rez.png")

    concatenare_orizontala(rez, im, (bgcolor[2],bgcolor[1],bgcolor[0])).save("Assets\\Objects\\rez.png")


cv.imshow("output", onlymaxarieminper)
cv.waitKey(0)
cv.destroyAllWindows()