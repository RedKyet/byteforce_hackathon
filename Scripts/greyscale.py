import cv2

"""def load_image(string path):
    image=cv2.imread('path')
    return image"""

img = cv2.imread('Assets\\Greyscale test\\rock.jpeg', 0)
 
cv2.imshow('Grayscaled', img)
cv2.waitKey(0)
cv2.destroyAllWindows()