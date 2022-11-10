import cv2 as cv
import math
import time
import os
import numpy as np

def obtinere_index(indeximg, line, column):
    return (((int(indeximg[line][column][2])) + ((int(indeximg[line][column][1])) << 8) + ((int(indeximg[line][column][0])) << 16)) - 1)