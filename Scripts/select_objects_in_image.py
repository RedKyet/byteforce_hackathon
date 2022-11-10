import cv2
from PIL import Image, ImageFont, ImageDraw

# read image without color
image = cv2.imread('Assets\\Greyscale test\\start.png')

original = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
#masked = cv
# 2.bitwise_and(original, original, mask=thresh)
#dilate = cv2.dilate(thresh, kernel, iterations=1)
#puts number on image
im1 = Image.open('ROI_0.png')
ac = Image.new('L', (im1.width, im1.height))


def adaugare_numar(a, num):
    title_font = ImageFont.truetype('Scripts\\fontu.ttf', 25)
    title_text = num
    image_editable = ImageDraw.Draw(a)
    image_editable.text([15, 15], title_text, fill="red", font=title_font)
    #ImageDraw.Draw.text(xy, text, fill=None, font=None, anchor=None, spacing=0, align=”left”)
    im1.save("result.png")

## 
# 
# # Find contours, obtain bounding box coordinates, and extract ROI
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

#cv2.imwrite('Assets\\Greyscale test\\rock_greyscaled.jpg',img)"""
image = cv2.imread('Assets\\Greyscale test\\start.png')
original = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
dilate = cv2.dilate(thresh, kernel, iterations=1)

# Find contours, obtain bounding box coordinates, and extract ROI
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]
    cv2.imwrite("ROI_{}.png".format(image_number), ROI)
    image_number += 1

cv2.imshow('image', image)
cv2.imshow('thresh', thresh)
cv2.imshow('dilate', dilate)
cv2.waitKey()