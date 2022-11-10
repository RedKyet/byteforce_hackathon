import cv2
import numpy as np
import os
 
dir = 'Assets\\Greyscale test\\rot\\'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
def medium(img):
    suma=0
    pixels=img.size
    dimensions = img.shape
    sizex=dimensions[0]
    sizey=dimensions[1]
    for x in range(0 , sizex):
        for y in range(0 , sizey):
            suma+=img[x][y]
    return suma/pixels
treshold=0.1
img = cv2.imread('Assets\\Greyscale test\\messi9.png',0)
dimensions = img.shape
sizex=int(dimensions[0]/2)
sizey=int(dimensions[1]/2)
for alpha in range(0, 360):
    rot=rotate_image(img,alpha)
   
    w=sizex
    half = w//2
  
    # this will be the first column
    left_part = rot[:, :half] 
  
    # [:,:half] means all the rows and
    # all the columns upto index half
  
    # this will be the second column
    right_part = rot[:, half:]  
    s = img[0:0, sizex:sizey]
    d = img[sizex:0, sizex*2:sizey*2]
    arial=medium(left_part)
    ariar=medium(right_part)
    if(abs(arial-ariar)<=treshold):
        print(str(arial)+" "+str(ariar))
        image = cv2.line(rot, (sizex,0), (sizex,sizey*2), (100,100,100), 5)
        cv2.imwrite('Assets\\Greyscale test\\rot\\messi'+str(alpha)+'.png',image)







medium(img)
cv2.imshow("img", img)
cv2.waitKey(0)