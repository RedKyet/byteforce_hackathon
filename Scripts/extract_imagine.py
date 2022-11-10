import cv2 as cv
import math
import time
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw


from adaugare_numar import adaugare_numar
from concatenare import concatenare_orizontala
from detect_culoare import get_color

##############################################################################################################

imgname = "Assets\\Greyscale test\\Gaina4.jpg"
binarythresh = 240
contrastthresh = 40

##############################################################################################################


img = cv.imread(imgname, cv.IMREAD_COLOR)
rows, cols, _ = img.shape
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
invbinaryimg = cv.threshold(grayimg, binarythresh, 255, cv.THRESH_BINARY_INV)[1]

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
cnts = cv.findContours(betterholesimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number = 0

blanc = np.zeros((1, 1, 3), np.uint8)

aux = Image.open('Assets\\Greyscale test\\Gaina4.jpg')
cul = get_color(aux)

cv.imwrite("Assets\\Objects\\rez.png".format(image_number), blanc)



for c in cnts:
    x,y,w,h = cv.boundingRect(c)
    #cv.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    element = img[y:y+h, x:x+w]
    cv.imwrite("Assets\\Objects\\element_{}.png".format(image_number), element)


    #creere imagnie cu numar
    im = Image.open("Assets\\Objects\\element_{}.png".format(image_number))
    adaugare_numar(im, image_number)
    image_number += 1


    
    rez = Image.open("Assets\\Objects\\rez.png")
    
    #originalImage = cv.imread(concatenare_orizontala(rez, im, (0, 0, 0)))
    #cv.imwrite("Assets\\Objects\\rez.png", concatenare_orizontala(rez, im, (0, 0, 0)))
    concatenare_orizontala(rez, im, cul).save("Assets\\Objects\\rez.png")
   # rez.save("Assets\\Objects\\rez.png")
    #creere imagine prin concatenare
    
    



cv.imshow("grayimg", grayimg)
cv.imshow("invbinaryimg", invbinaryimg)
cv.imshow("betterholesimg", betterholesimg)

cv.waitKey(0)

cv.destroyAllWindows()