import cv2 as cv

import numpy as np 
from PIL import Image

img = cv.imread('Assets\\Greyscale test\\fotografietest.png')
mask = cv.imread('Assets\\Greyscale test\\rock.jpeg', 0)

dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

"""
im1 = cv2.imread('Assets\\Greyscale test\\fotografietest.png')
im2 = cv2.imread('Assets\\Greyscale test\\rock.jpeg')


def get_concat_h(im1, im2):
    dst = Image.new('L', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('L', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

#
get_concat_h(im1, im1).save('Assets\Greyscale test\resulth.png')
get_concat_v(im1, im1).save('Assets\Greyscale test\resultv.png')

"""


