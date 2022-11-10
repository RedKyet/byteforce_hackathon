
# import image module from pillow
from PIL import Image, ImageFont, ImageDraw
import os
import fnmatch
import cv2 as cv

#numarat fisiere ROI
#  
im1 = Image.open('ROI_0.png')
ac = Image.new('L', (im1.width , im1.height))



def adaugare_numar (a, num):

    draw = ImageDraw.Draw(a)

    font = ImageFont.truetype('Scripts\\fontu.ttf', 25)

    text= str(num)

    draw.text((5, 5), text= text, fill="red", font=font, align="right")

    a.save("Assets\\Objects\\element_numar_{}.png".format(num))

'''

    title_font = ImageFont.truetype('Scripts\\fontu.ttf', 25)
    title_text = num
    image_editable = ImageDraw.Draw(a)
    image_editable.text([15, 15], title_text, fill="red", font=title_font)
    
    im1 = ImageDraw.Draw.text(xy, text, fill=None, font=None, anchor=None, spacing=0, align=”left”)
 #   cv.imwrite("Assets\\Objects\\element_numar_{}.png".format(num), im1)


'''
# open the image

'''
Image1 = Image.open('Assets\\Greyscale test\\start.png')

# make a copy the image so that
# the original image does not get affected
Image1copy = Image1.copy()
Image2 = Image.open('Assets\\Greyscale test\\rock.jpeg')
Image2copy = Image2.copy()

# paste image giving dimensions
Image1copy.paste(Image2copy, (0, 0))

# save the image
Image1copy.save('rez.png')
'''