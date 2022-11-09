import cv2
# read image without color
img = cv2.imread('Assets\\Greyscale test\\rock.jpeg',0)
cv2.imshow('Greyscaled',img)
cv2.waitKey(0)