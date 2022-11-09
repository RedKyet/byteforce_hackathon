import cv2
# read image without color
image = cv2.imread('Assets\\Greyscale test\\start.png')
"""cv2.imshow('Greyscaled',img)
cv2.waitKey(0)"""

original = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
#masked = cv2.bitwise_and(original, original, mask=thresh)
#dilate = cv2.dilate(thresh, kernel, iterations=1)

# Find contours, obtain bounding box coordinates, and extract ROI
#
#
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]
    cv2.imwrite("Assets\\Objects\\ROI_{}.png".format(image_number), ROI)
    image_number += 1
#
#
#

#cv2.imshow('masked', image)
cv2.imshow('image', image)
cv2.imshow('gray', gray)
cv2.imshow('thresh', thresh)

#masked = cv2.bitwise_and(image, image, mask=mask)

#cv2.imshow('dilate', dilate)
cv2.waitKey()

#cv2.imwrite('Assets\\Greyscale test\\rock_greyscaled.jpg',img)