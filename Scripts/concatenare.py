import cv2 as cv

import numpy as np
from PIL import Image


def concatenare_orizontala(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width,
                    max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst






'''

image = cv2.imread('Assets\\Greyscale test\\start.png')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
dilate = cv2.dilate(thresh, kernel, iterations=1)

# Find contours, obtain bounding box coordinates, and extract ROI
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number = 0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    ROI = original[y:y+h, x:x+w]
    cv2.imwrite("ROI_{}.png".format(image_number), ROI)
    image_number += 1
amp = 4

image = cv2.resize(image, (220*amp, 180*amp))
dilate = cv2.resize(dilate, (220*amp, 180*amp))

dst = cv2.inpaint(image, dilate, 3, cv2.INPAINT_TELEA)

cv2.imshow('image', image)

cv2.imshow('thresh', thresh)
cv2.imshow('dilate', dst)
cv2.waitKey(0)

"""
mask = cv.imread('Assets\\Greyscale test\\filtru.png', 0)


img = cv.resize(img, (220*amp, 180*amp))
mask = cv.resize(mask, (220*amp, 180*amp))


dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

cv.imshow('dst', img)
cv.waitKey(0)
cv.destroyAllWindows()


dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()


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


'''