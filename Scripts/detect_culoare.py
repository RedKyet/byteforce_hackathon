import cv2 as cv
import math
import time
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def get_color(im):

    pix = im.load()  # Get the RGBA Value of the a pixel of an image
    cul = pix[0, 0]   # Set the RGBA Value of the image (tuple)
    return cul


