from cmath import sqrt
import cv2
import imageio as iio
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import numpy as np
from PIL import Image
import imageio as iio
from skimage import filters
from skimage.color import rgb2gray  # only needed for incorrectly saved images
from skimage.measure import regionprops

img = cv2.imread('Assets\\Greyscale test\\redblue.jpg', 1)


width = img.shape[0]
height = img.shape[1]

i=6 
j=6


temp = img[i, j]
r1 = temp[0]
g1 = temp[1]
b1 = temp[2]

temp = img[width-i, height-j]
r2 = temp[0]
g2 = temp[1]
b2 = temp[2]

d = sqrt((r2-r1) ^ 2+(g2-g1) ^ 2+(b2-b1) ^ 2)
print(d)

for i in range(0, width//2):
    for j in range(0, height//2):

        temp = img[i][j]
        r1= temp[0]
        g1= temp[1]
        b1= temp[2]

        temp = img[width-i][height-j]
        r2 = temp[0]
        g2 = temp[1]
        b2 = temp[2]

        d = sqrt((r2-r1) ^ 2+(g2-g1) ^ 2+(b2-b1) ^ 2)







'''


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

print(w)c
mij = h/2
w = mij/2




def issym(img, procent, cul):
   
    h = img.height
'''
