import joblib

import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk
import cv2
import operator
from tensorflow import keras

SCALE = 0.33

def mask(img):
      lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l_channel, a, b = cv2.split(lab)
      clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10,10))
      cl = clahe.apply(l_channel)
      limg = cv2.merge((cl,a,b))
      enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
      gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

      proc = cv2.GaussianBlur(gray.copy(), (9, 9), 0)
      proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
      proc = cv2.bitwise_not(proc, proc)
      kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8) 
      proc = cv2.dilate(proc, kernel)

      img_cnt = img.copy()

      contours, h = cv2.findContours(proc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
      areas=[]
      for  cnt in contours:
        areas.append([cv2.contourArea(cnt), cnt])

      areas.sort(key=lambda x:x[0],reverse=True)
      
      sudoku=[areas[0]]
      for i in range(2, len(areas)):
        if (areas[i][0]>0):
          if (areas[i][0]/areas[1][0]>=0.85): sudoku.append(areas[i])
          else: break
      
      contours=[]
      for i in range(len(sudoku)): contours.append(sudoku[i][1])
      
      img_cnt[:] = (0, 0, 0)
      cv2.fillPoly(img_cnt, pts =contours, color=(255,255,255))
      img_cnt=cv2.cvtColor(img_cnt, cv2.COLOR_BGR2GRAY)
      return img_cnt, contours
def crop(img, contour):
        def en(a=1): 
          return enumerate([pt[0][0]+a*pt[0][1] for pt in contour])
        def dist(p1, p2): 
          return np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))  
        
        tl=contour[min(en(1), key=operator.itemgetter(1))[0]][0]
        tr=contour[max(en(-1), key=operator.itemgetter(1))[0]][0]
        br=contour[max(en(1), key=operator.itemgetter(1))[0]][0]
        bl=contour[min(en(-1), key=operator.itemgetter(1))[0]][0]
        
        src = np.array([tl, tr, br, bl], dtype='float32')

        side = max([ dist(br, tr), dist(tl, bl), dist(br, bl), dist(tl, tr)])

        dst = np.array([[0, 0], 
                        [side - 1, 0], 
                        [side - 1, side - 1], 
                        [0, side - 1]], dtype='float32')
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, m, (int(side), int(side)))
def scale(img, size):
    h, w = img.shape[:2]

    def centre_pad(length): return int((size - length) / 2), int((size - length) / 2)+1*(length % 2 != 0)
       
    ratio = (size-4) / max(h, w)
    w, h = int(ratio*w), int(ratio*h)
    img = cv2.resize(img, (w, h))
    if h > w:    
        l_pad, r_pad = centre_pad(w)
        img = cv2.copyMakeBorder(img, 2, 2, l_pad, r_pad, cv2.BORDER_CONSTANT, None, 0)
    else:
        t_pad, b_pad = centre_pad(h)
        img = cv2.copyMakeBorder(img, t_pad, b_pad, 2, 2, cv2.BORDER_CONSTANT, None, 0)

    
    return cv2.resize(img, (size, size))
def find_largest_feature(inp_img, scan_tl, scan_br):
    img = inp_img.copy()
    height, width = img.shape[:2]
    max_area = 0
    seed_point = (None, None)
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height: 
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)

    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:
                cv2.floodFill(img, mask, (x, y), 0)

            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point
def extract_digit(img, rect, size):
    
    digit = img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    bbox= find_largest_feature(digit, [margin, margin], [w - margin, h - margin])[1]
    digit = digit[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0: return scale(digit, size)
    else: return np.zeros((size, size), np.uint8)
def get_digits(img, squares, size):
    digits = []
    img_copy=img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    proc = cv2.bitwise_not(proc, proc)
    for square in squares:
        digits.append(extract_digit(proc, square, size))
    return digits
def one_digit(img, m):
        side_y=img.shape[0]/9
        side_x=img.shape[1]/9
        image=[]
        for i in range(9):
            for j in range(9):
                image.append(img[int(side_x*i):int(side_x*(i+1)), int(side_y*j):int(side_y*(j+1))])
        return image[m]

def preprocess_table(table):
    x= []
    side = table.shape[:1][0]/9
    for j in range(9):
        for i in range(9):
            x.append(((i * side, j * side), ((i + 1) * side, (j + 1) * side)))
    
    digits = get_digits(table, x, 28)
    rows = []
    with_border = [cv2.copyMakeBorder(z.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 255) for z in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    return np.concatenate(rows)
    
def digits_answer(proc, clf):
    sudoku_digits=[]
    for i in range(81):
      digit_img=one_digit(proc, i)
      image = digit_img[1:29,1:29]
      image = cv2.resize(image, (28,28))
      kernel = np.ones((2,2))
      image = cv2.dilate(image, kernel, iterations=2)
      image = cv2.erode(image, kernel, iterations=2)
      # print(image.sum())
      if image.sum() > 10000:
          features_test = np.array([image.ravel()])
          sudoku_digit= clf.predict(features_test)
          sudoku_digits.append(sudoku_digit[0])
      else:
          
          sudoku_digits.append(-1)
      
    sudoku_digits=np.array(sudoku_digits)
    return sudoku_digits.reshape(9, 9)

def predict_image(image):
    img_cnt, contours=mask(image)
    mask_pred = img_cnt
    
    
    cropped=crop(image, contours[0])
    
    clf = joblib.load('/autograder/submission/model.joblib')
    proc=preprocess_table(cropped)
    sudoku_digits = [np.int16(digits_answer(proc, clf))]
   
    return mask_pred, sudoku_digits
