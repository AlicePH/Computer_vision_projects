from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from skimage.feature import match_template
from scipy.spatial.distance import cdist
import scipy.stats as st
COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}


def predict_image(img: np.ndarray):
    def rotate_image(image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result
    def number_trains(image, color):
      img_for_template= cv2.imread('train/all.jpg')
      img_for_template=cv2.cvtColor(img_for_template, cv2.COLOR_BGR2RGB)
      
      count=0
      if (color=='blue'):

        pass
      if (color=='green'):
        img_all=img_for_template[790:815, 2080:2150]
        #img_all=cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        background = np.median(HLS, axis=(0,1))

        mask = np.mean(np.abs(HLS-background), axis=-1) < 25
        img_new = image.copy()
        img_new[mask] = 0

        img_orange_copy=getMask(img_new, "green")
        image_1=img_orange_copy
        level=0.34
      if (color=='black'):
        img_all=img_for_template[1200:1250, 1100:1200]
        img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_1=img

        level=0.45
        
      if (color=='yellow'):
        img_all=img_for_template[910:1020, 810:850]
        #img_all=cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        background = np.median(HLS, axis=(0,1))

        mask = np.mean(np.abs(HLS-background), axis=-1) < 25
        img_new = image.copy()
        img_new[mask] = 0

        img_orange_copy=getMask(img_new, "cyan")
        image_1=img_orange_copy
        level=0.4

      if (color=='red'):
        img_all=img_for_template[1335:1380, 1800:1900]
        #img_all=cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        background = np.median(HLS, axis=(0,1))

        mask = np.mean(np.abs(HLS-background), axis=-1) < 25
        img_new = image.copy()
        img_new[mask] = 0

        img_orange_copy=getMask(img_new, "blue")
        image_1=img_orange_copy
        level=0.5
        
      
    
      img_all=cv2.cvtColor(img_all, cv2.COLOR_RGB2GRAY) #шаблон
      image_1=cv2.cvtColor(image_1, cv2.COLOR_HLS2RGB) #основная картинка
      image_1=cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
      x_all=match_template(image_1, img_all, 1)
      y_all=label(x_all>level, 2, 1)
      num=y_all[1]*(y_all[1]>15)
      # x_bad=match_template(image_1, img_bad, 1)
      # y_bad=label(x_bad>level, 2, 1)
      
      # return (y_all[1]-y_bad[1])*((y_all[1]-y_bad[1])>0)
      return num
    def white (image, array):

      for i in range(image.shape[0]):
        for j in range(image.shape[1]):
          if (list(image[i][j]) !=array): image[i][j]=[255, 255, 255]
      return image

    def city_center(image):
      img=cv2.imread('train/all.jpg')
      img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img=img[1190:1245, 3477:3532]
      image_1=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      x=match_template(image_1, img, 1)
      y=label(x> 0.55, 2, 1)
      coordinates=[]
      for i in range(y[1]):
        coordinates.append(np.mean(np.argwhere(y[0] == i+1), 0))
      
      coordinates=np.int64(np.round(coordinates))

      return coordinates
    def how_many(image, color):
      img_new = image.copy()
      img_orange_copy=getMask(img_new, color) # orange for blue trains
        
      gray = cv2.cvtColor(img_orange_copy, cv2.COLOR_RGB2GRAY)
      blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
      divide = cv2.divide(gray, blur, scale=255)
      thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
      morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

      contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      areas=[]
      for cnt in contours: 
        if (cv2.contourArea(cnt)>3500): areas.append(cv2.contourArea(cnt))
        

      return len(areas)
    def detectCirclesWithDp(frame, dp=1):
      blurred = cv2.medianBlur(frame, 25)
      grayMask = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
      return cv2.HoughCircles(grayMask, cv2.HOUGH_GRADIENT, dp, 40, param1=10, param2=30, minRadius=20, maxRadius=70)

    def getROI(frame, x, y, r):
        return frame[int(y-r/2):int(y+r/2), int(x-r/2):int(x+r/2)]

    COLOR_NAMES = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']

    COLOR_RANGES_HSV = {
        "red": [(0, 50, 10), (10, 255, 255)],
        "orange": [(10, 50, 10), (25, 255, 255)],
        "yellow": [(25, 50, 10), (35, 255, 255)],
        "green": [(35, 50, 10), (80, 255, 255)],
        "cyan": [(80, 50, 10), (100, 255, 255)],
        "blue": [(100, 50, 10), (130, 255, 255)],
        "purple": [(130, 50, 10), (170, 255, 255)],
    }

    def getMask(frame, color):
        blurredFrame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

        colorRange = COLOR_RANGES_HSV[color]
        lower = np.array(colorRange[0])
        upper = np.array(colorRange[1])

        colorMask = cv2.inRange(hsvFrame, lower, upper)
        colorMask = cv2.bitwise_and(blurredFrame, blurredFrame, mask=colorMask)

        return colorMask

    def is_contour_bad(c):
          peri = cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, 0.02 * peri, True)
          return not len(approx) == 4

    

    # img_n_copy=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # background = np.median(HLS, axis=(0,1))
    
    # mask = np.mean(np.abs(HLS-background), axis=-1) < 25
    # img_new = img.copy()
    # img_new[mask] = 0
    
    # img_orange_copy=getMask(img_new, "blue")
    # circles=detectCirclesWithDp(img_orange_copy)
    # city_centers=circles[:, :, :-1]
    # city_centers=np.int64(city_centers)
    # img_shape_base=[2546, 3846, 3]
    # x_coeff=img.shape[0]/img_shape_base[0]
    # y_coeff=img.shape[1]/img_shape_base[1]

    n_trains_blue=0
    n_trains_green=number_trains(img, 'green')
    n_trains_yellow=number_trains(img, 'yellow')
    n_trains_red=number_trains(img, 'red')
    n_trains_black=number_trains(img, 'black')
    # n_trains = {'blue': n_trains_blue, 'green': n_trains_green, 'black': 15, 'yellow': n_trains_yellow, 'red': 15}
    # scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
    # x=[[157, 2151], [227, 648], [236, 3200], [260, 2584], [436, 1778], [587, 2287], [672, 3551], [745, 2866], [760, 3242], [766, 887], [781, 1233], [812, 1524], [836, 2460], [866, 1878], [948, 1151], [1024, 3009], [1063, 1469], [1090, 848], [1190, 521], [1218, 3503], [1218, 1672], [1248, 1021], [1281, 2081], [1369, 2254], [1409, 3651], [1448, 1433], [1563, 1733], [1609, 2036], [1624, 2800], [1678, 3303], [1736, 3630], [1781, 1333], [1800, 787], [1821, 2339], [1851, 2578], [1887, 1760], [1981, 2066], [2075, 2987], [2096, 427], [2133, 827], [2181, 175], [2193, 3563], [2266, 2515], [2278, 3269], [2363, 2827], [2363, 1893], [2375, 420]]
    # for i in range(len(x)): x[i][0]*=x_coeff
    # for i in range(len(x)): x[i][1]*=y_coeff

    n_trains = {'blue': n_trains_blue, 'green': n_trains_green, 'black': n_trains_black, 'yellow': n_trains_yellow, 'red': n_trains_red}
    scores = {'blue': 1.5*n_trains_blue, 'green': n_trains_green*1.5, 'black': n_trains_black*1.5, 'yellow': n_trains_yellow*1.5, 'red': n_trains_red*1.5}
    return city_center(img), n_trains, scores
