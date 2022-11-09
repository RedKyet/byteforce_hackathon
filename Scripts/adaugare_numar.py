from PIL import Image


img = Image.open('Assets\\Greyscale test\\start.png')

img_w, img_h = img.size
background = Image.new('L', (1440, 900), (255, 255, 255, 255))
bg_w, bg_h = background.size
offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
background.paste(img, offset)
background.save('out.png')
