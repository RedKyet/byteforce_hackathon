from cmath import sqrt
import cv2 as cv
import math
import os
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def magic(imagepath: str):

    imgname = imagepath
    path = imgname.rsplit('\\', 1)[0]
    path = path + '\\'
    photofolder = path + "photos\\"
    fontfilepath = "Scripts\\fontu.ttf"
    binarythresh = 251
    epsilonarie = 5.0
    epsilonper = 1
    epsilonbright = 0.005
    epsilondark = 0.005
    symmetrystep = 0.017
    symmetrythresh = 0.2
    lowfilldiff = 5
    highfilldiff = 5
    finalsymmetrythresh = 0.998
    smallthresh = 10

    ##########################################################################################################

    if not os.path.exists(path+"photos"):
        os.mkdir(path+"photos")
        
    def obtinere_index(indeximg, line, column):
        return (((int(indeximg[line][column][2])) + ((int(indeximg[line][column][1])) << 8) + ((int(indeximg[line][column][0])) << 16)) - 1)


    '''--------------------ADAUGAT-----------NUMAR--------------------------------- '''

    def adaugare_numar (a, num):
        draw = ImageDraw.Draw(a)
        font = ImageFont.truetype(fontfilepath, 25)
        text= str(num+1)
        draw.text((5, 5), text= text, fill="red", font=font, align="right")
        a.save(photofolder+"{}_withnumber.png".format(num))

    def concatenare_orizontala(im1, im2, color=(0, 0, 0)):
        dst = Image.new('RGB', (im1.width + im2.width,
                        max(im1.height, im2.height)), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    ##########################################################################################################

    originalimg = cv.imread(imgname, cv.IMREAD_UNCHANGED)
    rows, cols, _ = originalimg.shape
    if len(originalimg[0][0]) == 4:
        for i in range(rows):
            for j in range(cols):
                if originalimg[i][j][3] < 255:
                    originalimg[i][j][0] = 255
                    originalimg[i][j][1] = 255
                    originalimg[i][j][2] = 255
        img = cv.cvtColor(originalimg, cv.COLOR_BGRA2BGR)
    else:
        img = originalimg.copy()

    # turn background to white

    bgbb = (0, 0, 0, 0)
    bgbbarea = -1
    bgcolor = (0, 0, 0)

    wbimg = img.copy()
    rect = cv.floodFill(wbimg, None, (0, 0), (255, 255, 255), loDiff=lowfilldiff, upDiff=highfilldiff, flags=4+cv.FLOODFILL_FIXED_RANGE)[3]
    rectarea = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)
    if rectarea > bgbbarea:
        bgbb = rect
        bgbbarea = rectarea
        bgcolor = img[0][0]
        finalwbimg = wbimg.copy()

    wbimg = img.copy()
    rect = cv.floodFill(wbimg, None, (0, rows-1), (255, 255, 255), loDiff=lowfilldiff, upDiff=highfilldiff, flags=4+cv.FLOODFILL_FIXED_RANGE)[3]
    rectarea = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)
    if rectarea > bgbbarea:
        bgbb = rect
        bgbbarea = rectarea
        bgcolor = img[rows-1][0]
        finalwbimg = wbimg.copy()

    wbimg = img.copy()
    rect = cv.floodFill(wbimg, None, (cols-1, 0), (255, 255, 255), loDiff=lowfilldiff, upDiff=highfilldiff, flags=4+cv.FLOODFILL_FIXED_RANGE)[3]
    rectarea = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)
    if rectarea > bgbbarea:
        bgbb = rect
        bgbbarea = rectarea
        bgcolor = img[0][cols-1]
        finalwbimg = wbimg.copy()

    wbimg = img.copy()
    rect = cv.floodFill(wbimg, None, (cols-1, rows-1), (255, 255, 255), loDiff=lowfilldiff, upDiff=highfilldiff, flags=4+cv.FLOODFILL_FIXED_RANGE)[3]
    rectarea = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)
    if rectarea > bgbbarea:
        bgbb = rect
        bgbbarea = rectarea
        bgcolor = img[rows-1][cols-1]
        finalwbimg = wbimg.copy()

    bgcolor = [int(s) for s in bgcolor]

    for i in range(rows):
        for j in range(cols):
            if finalwbimg[i][j][0] == bgcolor[0] and finalwbimg[i][j][1] == bgcolor[1] and finalwbimg[i][j][2] == bgcolor[2]:
                finalwbimg[i][j] = [255, 255, 255]

    wbimg = finalwbimg.copy()

    # get object contours

    grayimg = cv.cvtColor(wbimg, cv.COLOR_BGR2GRAY)
    invbinaryimg = cv.threshold(grayimg, binarythresh, 255, cv.THRESH_BINARY_INV)[1]

    contours = cv.findContours(invbinaryimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    contoursbinaryimg = invbinaryimg.copy()
    cv.drawContours(contoursbinaryimg, contours, -1, color=(255, 255, 255), thickness=cv.FILLED)

    # sort contours by area in decreasing order

    contours = list(contours)
    contours.sort(key=lambda c:cv.contourArea(c), reverse=True)

    # eliminate false objects (lmao)

    while len(contours) > 0 and cv.contourArea(contours[len(contours)-1]) <= smallthresh:
        contours.pop()

    # give indexes to objects

    indexedimg = np.zeros((rows, cols, 3), np.uint8)

    for i in range(len(contours)):
        cr = ((i+1) & 255)
        cg = (((i+1) >> 8) & 255)
        cb = (((i+1) >> 16) & 255)
        cv.drawContours(indexedimg, [contours[i]], -1, color=(cb, cg, cr), thickness=cv.FILLED)

    # make lists with object properties

    objectprops = [{} for i in range(len(contours))]

    intensities = [0 for i in range(len(contours))]
    nrpixels = [0 for i in range(len(contours))]
    objectareas = [0 for i in range(len(contours))]
    objectperimeters = [0 for i in range(len(contours))]
    objectcentroidrow = [0 for i in range(len(contours))]
    objectcentroidcol = [0 for i in range(len(contours))]
    objectsymmetryscore = [0 for i in range(len(contours))]
    objectsymmetryaxis = [0 for i in range(len(contours))]

    # calculate object pixel number and intensities (RSUM + BSUM + GSUM)

    for i in range(rows):
        for j in range(cols):
            ind = obtinere_index(indexedimg, i, j)
            if ind >= 0 and invbinaryimg[i][j] == 255:
                nrpixels[ind] += 1
                intensities[ind] += (int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2]))

    # generate image with transparent background

    alphaimg = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    for i in range(rows):
        for j in range(cols):
            if invbinaryimg[i][j] == 255:
                alphaimg[i][j][3] = 255
            else:
                alphaimg[i][j][3] = 0

    # keep max area and min perimeter objects

    maxarie = -1
    for i in range(len(contours)):
        if cv.contourArea(contours[i]) > maxarie:
            maxarie = cv.contourArea(contours[i])
        objectareas[i] = cv.contourArea(contours[i])

    minper = 9999999
    for i in range(len(contours)):
        if cv.arcLength(contours[i], True) < minper:
            minper = cv.arcLength(contours[i], True)
        objectperimeters[i] = cv.arcLength(contours[i], True)

    onlymaxarieminper = img.copy()

    ok = False
    for i in range(len(contours)):
        currarie = cv.contourArea(contours[i])
        currper = cv.arcLength(contours[i], True)
        if maxarie-currarie <= epsilonarie and currper-minper <= epsilonper:
            ok = True

    if not ok:
        for i in range(len(contours)):
            currarie = cv.contourArea(contours[i])
            currper = cv.arcLength(contours[i], True)
            if maxarie-currarie > epsilonarie and currper-minper > epsilonper:
                cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=cv.FILLED)
                cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=8)
    else:
        for i in range(len(contours)):
            currarie = cv.contourArea(contours[i])
            currper = cv.arcLength(contours[i], True)
            if maxarie-currarie > epsilonarie or currper-minper > epsilonper:
                cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=cv.FILLED)
                cv.drawContours(onlymaxarieminper, [contours[i]], -1, color=bgcolor, thickness=8)

    # keep max and min brightness

    maxbrightness = -1.0
    minbrightness = 10250.0
    for i in range(len(contours)):
        bright = intensities[i] / nrpixels[i]
        if bright > maxbrightness:
            maxbrightness = bright
        if bright < minbrightness:
            minbrightness = bright

    onlybrightanddark = img.copy()

    for i in range(len(contours)):
        bright = intensities[i] / nrpixels[i]
        if bright-minbrightness > epsilondark and maxbrightness-bright > epsilonbright:
            cv.drawContours(onlybrightanddark, [contours[i]], -1, color=bgcolor, thickness=cv.FILLED)
            cv.drawContours(onlybrightanddark, [contours[i]], -1, color=bgcolor, thickness=8)

    # update dictionary

    for i in range(len(contours)):
        objectprops[i]['intensities'] = intensities[i]
        objectprops[i]['nrpixels'] = nrpixels[i]
        objectprops[i]['area'] = objectareas[i]
        objectprops[i]['perimeter'] = objectperimeters[i]
        objectprops[i]['brightness'] = intensities[i]/nrpixels[i]

    # centroid calculation

    for i in range(rows):
        for j in range(cols):
            ind = obtinere_index(indexedimg, i, j)
            if ind >= 0 and invbinaryimg[i][j] == 255:
                objectcentroidcol[ind] += j
                objectcentroidrow[ind] += i

    for i in range(len(contours)):
        objectcentroidcol[i] /= nrpixels[i]
        objectcentroidrow[i] /= nrpixels[i]
        objectprops[i]['centroidrow'] = objectcentroidrow[i]
        objectprops[i]['centroidcol'] = objectcentroidcol[i]
        

    # sort objects in decreasing order of areas AND calculate symmetry

    image_number=0
    blanc = np.zeros((1, 1, 3), np.uint8)
    cv.imwrite(photofolder+"\\longlong.png", blanc)

    onlyasymmetricalimg = img.copy()

    for c in range(len(contours)):

        # creare imagine cu numar

        x, y, w, h = cv.boundingRect(contours[c])
        element = alphaimg[y:y+h, x:x+w]

        for i in range(h):
            for j in range(w):
                ind = obtinere_index(indexedimg, y+i, x+j)
                if ind is not c:
                    element[i][j][3] = 0
                    element[i][j][0] = bgcolor[0]
                    element[i][j][1] = bgcolor[1]
                    element[i][j][2] = bgcolor[2]
        
        objectprops[c]['boundingbox'] = [y, x, y+h-1, x+w-1]

        cv.imwrite(photofolder+"{}.png".format(c), element)
        im = Image.open(photofolder+"{}.png".format(c))

        objectprops[c]['filename'] = (photofolder+"{}.png".format(c)).split('\\', 1)[1]

        # calculare simetrie

        binaryelement = invbinaryimg[y:y+h, x:x+w]
        for i in range(h):
            for j in range(w):
                ind = obtinere_index(indexedimg, y+i, x+j)
                if ind != c:
                    binaryelement[i][j] = 0
        
        cv.imwrite("{}_binarything.png", binaryelement)

        centroid = (objectcentroidrow[c]-y, objectcentroidcol[c]-x)
        partialsumrow = [[0 for j in range(cols)] for i in range(rows)]
        partialsumcol = [[0 for j in range(cols)] for i in range(rows)]

        for i in range(h):
            j = 0
            if binaryelement[i][j] == 255:
                partialsumrow[i][j] = 1
            for j in range(1, w):
                partialsumrow[i][j] = partialsumrow[i][j-1]
                if binaryelement[i][j] == 255:
                    partialsumrow[i][j] += 1
        
        for j in range(w):
            i = 0
            if binaryelement[i][j] == 255:
                partialsumcol[i][j] = 1
            for i in range(1, h):
                partialsumcol[i][j] = partialsumcol[i-1][j]
                if binaryelement[i][j] == 255:
                    partialsumcol[i][j] += 1
        
        # normalizare symmetrystep

        if h+w <= 180:
            symmetrystep = 0.054
        elif h+w <= 360:
            symmetrystep = 0.036
        else:
            symmetrystep = 0.018
        
        # cazul linie orizontala

        theta = 0
        m = 0

        legit = 0

        # y = m * (x - centroid[1]) + centroid[0]
        # x = (y - centroid[0]) / m + centroid[1]
        area1 = 0
        area2 = 0
        if w <= h:
            for j in range(w):
                i = int(centroid[0])
                if i >= 0 and i < h:
                    if i > 0:
                        area1 += partialsumcol[i-1][j]
                    if i < h-1:
                        area2 += (partialsumcol[h-1][j] - partialsumcol[i][j])
                elif i < 0:
                    area2 += partialsumcol[h-1][j]
                else:
                    area1 += partialsumcol[h-1][j]
        else:
            for i in range(h):
                if i < int(centroid[0]):
                    area1 += partialsumrow[i][w-1]
                elif i > int(centroid[0]):
                    area2 += partialsumrow[i][w-1]

        
        if area1+area2 == 0 or (abs(area1-area2)-20) / (area1+area2) < symmetrythresh:
            #iterat prin fiecare punct si calculat coordonatele punctului simetric(ib, jb)

            legit = 0
            for i in range (0, h):
                for j in range (0, w):
                    if binaryelement[i][j] == 255:
                        jb = j
                        ib = int(centroid[0]+(centroid[0]-i))

                        if ib >= 0 and ib < h and jb >= 0 and jb < w and binaryelement[i][j] == binaryelement[ib][jb]:
                            legit = legit + 1
            legit = (legit + objectperimeters[c]//2) / nrpixels[c]
        
        if legit > objectsymmetryscore[c]:
            objectsymmetryscore[c] = legit
            objectsymmetryaxis[c] = m
        
        # cazul linie verticala

        theta = math.pi/2
        m = math.tan(math.pi/2)

        legit = 0

        # y = m * (x - centroid[1]) + centroid[0]
        # x = (y - centroid[0]) / m + centroid[1]
        area1 = 0
        area2 = 0
        if w <= h:
            for j in range(w):
                if j < int(centroid[1]):
                    area1 += partialsumcol[h-1][j]
                elif j > int(centroid[1]):
                    area2 += partialsumcol[h-1][j]
        else:
            for i in range(h):
                j = int(centroid[1])
                if j >= 0 and j < w:
                    if j > 0:
                        area1 += partialsumrow[i][j-1]
                    if j < w-1:
                        area2 += (partialsumrow[i][w-1] - partialsumrow[i][j])
                elif j < 0:
                    area2 += partialsumrow[i][w-1]
                else:
                    area1 += partialsumrow[i][w-1]
                
    
        if area1+area2 == 0 or (abs(area1-area2)-20) / (area1+area2) < symmetrythresh:
            #iterat prin fiecare punct si calculat coordonatele punctului simetric(ib, jb)

            legit = 0
            for i in range (0, h):
                for j in range (0, w):
                    if binaryelement[i][j] == 255:
                        ib = i
                        jb = int(centroid[1] + (centroid[1]-j))

                        if ib >= 0 and ib < h and jb >= 0 and jb < w and binaryelement[i][j] == binaryelement[ib][jb]:
                            legit = legit + 1
            legit = (legit + objectperimeters[c]//2) / nrpixels[c]
        
        if legit > objectsymmetryscore[c]:
            objectsymmetryscore[c] = legit
            objectsymmetryaxis[c] = m
        
        # cazuri generale

        theta = symmetrystep
        while theta < math.pi:

            m = math.tan(theta)
            print(theta)

            legit = 0

            if m != 0:

                # y = m * (x - centroid[1]) + centroid[0]
                # x = (y - centroid[0]) / m + centroid[1]
                area1 = 0
                area2 = 0
                if w <= h:
                    for j in range(w):
                        i = int(m * (j - centroid[1]) + centroid[0])
                        if i >= 0 and i < h:
                            if i > 0:
                                area1 += partialsumcol[i-1][j]
                            if i < h-1:
                                area2 += (partialsumcol[h-1][j] - partialsumcol[i][j])
                        elif i < 0:
                            area2 += partialsumcol[h-1][j]
                        else:
                            area1 += partialsumcol[h-1][j]
                else:
                    for i in range(h):
                        j = int((i - centroid[0]) / m + centroid[1])
                        if j >= 0 and j < w:
                            if j > 0:
                                area1 += partialsumrow[i][j-1]
                            if j < w-1:
                                area2 += (partialsumrow[i][w-1] - partialsumrow[i][j])
                        elif j < 0:
                            area2 += partialsumrow[i][w-1]
                        else:
                            area1 += partialsumrow[i][w-1]

                if area1+area2 == 0 or (abs(area1-area2)-20) / (area1+area2) < symmetrythresh:
                    #iterat prin fiecare punct si calculat coordonatele punctului simetric(ib, jb)

                    legit = 0
                    m2 = (-1) / m
                    for i in range (0, h):
                        for j in range (0, w):
                            if binaryelement[i][j] == 255:
                                x = (i-m2*j+m*centroid[1]-centroid[0])/(m-m2)
                                y = m*x+(-m*centroid[1]+centroid[0])
                                jb = int(x+(x-j))
                                ib = int(y+(y-i))

                                if ib >= 0 and ib < h and jb >= 0 and jb < w and binaryelement[i][j] == binaryelement[ib][jb]:
                                    legit = legit + 1
                    legit = (legit + objectperimeters[c]//2) / nrpixels[c]
            
            if legit > objectsymmetryscore[c]:
                objectsymmetryscore[c] = legit
                objectsymmetryaxis[c] = m
            
            theta += symmetrystep
        
        objectprops[c]['symmetry'] = objectsymmetryscore[c]
        objectprops[c]['symmetryaxis'] = objectsymmetryaxis[c]
        objectprops[c]['contourlength'] = len(contours[c])

        if objectsymmetryscore[c] >= finalsymmetrythresh:
            cv.drawContours(onlyasymmetricalimg, [contours[c]], -1, color=bgcolor, thickness=cv.FILLED)
            cv.drawContours(onlyasymmetricalimg, [contours[c]], -1, color=bgcolor, thickness=8)
        
        # concatenare

        adaugare_numar(im, c)
        objectprops[c]['withnumbername'] = (photofolder+"{}_withnumber.png".format(c)).split('\\', 1)[1]
        rez = Image.open(photofolder+"longlong.png")
        concatenare_orizontala(rez, im, (bgcolor[2],bgcolor[1],bgcolor[0])).save(photofolder+"longlong.png")
    
    objectpropsdict = {}
    for i in range(len(contours)):
        objectpropsdict[str(i)] = objectprops[i]

    with open(path+"isdone.txt", "w") as isdonefile:
        isdonefile.write("100")
    objectpropsjson = json.dumps(objectprops, indent=2)
    objectpropsjsononeline = objectpropsjson.strip('\r\n\t ')
    with open(path+"data.txt", "w") as objectpropsfile:
        objectpropsfile.write(objectpropsjson)
    
    cv.imwrite(photofolder+"asymmetrical.png", onlyasymmetricalimg)
    cv.imwrite(photofolder+"bright_dark.png", onlybrightanddark)
    cv.imwrite(photofolder+"maxarea_minperimeter.png", onlymaxarieminper)

    return objectpropsdict

magic("Website\\static\\users\\mf8SaEchO8o\\cake.png")