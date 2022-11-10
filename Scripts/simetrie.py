import cv2

import numpy as np
from PIL import Image


import cv2 as cv

import numpy as np
from PIL import Image


img = cv.imread('Assets\\Greyscale test\\start.png', 0)
rows = img.shape[0]
columns = img.shape[1]

cv.imshow("maf iubeste javascript",img)
cv2.waitKey()
#axis('on', 'image')
sumgx = 0
sumgy = 0
sumg = 0




for col in range(0,columns): 
    for row in range(0, rows):
        gl = img[col][row]
        sumg = sumg + gl
        sumgx = sumgx + col * gl
        sumgy = sumgy + row * gl

xCOG = sumgx / sumg
yCOG = sumgy / sumg

cv.xline(xCOG, 'LineWidth', 2, 'Color', 'r')
cv.yline(yCOG, 'LineWidth', 2, 'Color', 'r')
caption = cv.sprintf('Center of Gravity: (%.2f, %.2f)', xCOG, yCOG)
cv.title(caption, 'FontSize', 18)

w = img.width
h = img.height

print(w)
mij = h/2
w = mij/2




def issym(img, procent, cul):
   
    h = img.height
