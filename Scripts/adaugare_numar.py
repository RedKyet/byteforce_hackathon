
# import image module from pillow
from PIL import Image, ImageFont, ImageDraw
import os
import fnmatch


#numarat fisiere ROI 
im1 = Image.open('ROI_0.png')
ac = Image.new('L', (im1.width , im1.height))
def adaugare_numar (a, num):
    title_font = ImageFont.truetype('Scripts\\fontu.ttf', 25)
    title_text = num
    image_editable = ImageDraw.Draw(a)
    image_editable.text([15, 15], title_text, fill="red", font=title_font)
    #ImageDraw.Draw.text(xy, text, fill=None, font=None, anchor=None, spacing=0, align=”left”)
    im1.save("result.png")


# open the image

adaugare_numar(im1, "7")
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