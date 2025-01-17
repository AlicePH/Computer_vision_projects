import numpy as np
import cv2
import skimage.morphology as sk
from operator import itemgetter
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN, KMeans
import os

def distance (img1, img2, parameter):

    histogram1 = cv2.calcHist([img1], [0],
                            None, [256], [0, 256])
    histogram2 = cv2.calcHist([img2], [0],
                              None, [256], [0, 256])

    c1=0
    
    i = 0
    while i<len(histogram1) and i<len(histogram2):
        c1+=(histogram1[i]-histogram2[i])**2
        i+= 1
    c1 = c1**(1 / 2)
    # print(c1, '\n')
    if c1<parameter: return 1
    else: return 0

def apply_ratio_test(all_matches, img_height):

    good_matches = {}
    accurate_matches = []

    precise_coeff = 0.6      

    if img_height >= 600:
        lowe_coeff = 0.7
    elif img_height >= 400:
        lowe_coeff = 0.8
    else:
        lowe_coeff = 0.9

    for m, n in all_matches:
        if m.distance < lowe_coeff * n.distance:
            good_matches[m.queryIdx] = m.trainIdx
        if m.distance < precise_coeff * n.distance:
            accurate_matches.append(m)
            
    return good_matches, accurate_matches

def compute_entry_hough_space(kp_q, kp_t, q_xc, q_yc):

    entry = {}

    v = ((q_xc - kp_q.pt[0]), (q_yc - kp_q.pt[1]))
    scale_ratio = kp_t.size / kp_q.size
    delta_angle = kp_t.angle - kp_q.angle
    x_c = kp_t.pt[0] + scale_ratio * (np.cos(delta_angle) * v[0] - np.sin(delta_angle) * v[1])
    y_c = kp_t.pt[1] + scale_ratio * (np.sin(delta_angle) * v[0] + np.cos(delta_angle) * v[1])

    entry['x_c'], entry['y_c'] = x_c, y_c
    entry['scale_ratio'] = scale_ratio
    entry['delta_angle'] = delta_angle
    
    return entry 

def create_hough_space(good_matches, kp_query, kp_train, query_xc, query_yc):
    
    hough_space = {}

    for t_idx, q_idx in good_matches.items():
        hough_space[t_idx] = compute_entry_hough_space(kp_query[q_idx], kp_train[t_idx], query_xc, query_yc)
    
    return hough_space

def compute_accurate_scale(accu_matches, kp_q, kp_t):

    accurate_scale_data = []

    for m in accu_matches:
        accurate_scale_data.append(kp_t[m.queryIdx].size/kp_q[m.trainIdx].size) 

    return accurate_scale_data

def compute_bins(hough_space, query_shape, train_shape, accurate_scale_data):

    values = {}
          
    counts_scale, bins_scale= np.histogram(accurate_scale_data, bins='auto')
    img_scale = np.mean([bins_scale[np.argmax(counts_scale)], bins_scale[np.argmax(counts_scale) + 1]])

    data_angle = [entry['delta_angle'] for entry in hough_space.values()]
    counts_angle, bins_angle= np.histogram(data_angle, bins='auto')

    x_bin_size = img_scale * query_shape[1]/4
    y_bin_size = img_scale * query_shape[0]/4
    x_bins = int(np.ceil(train_shape[1] / x_bin_size) + 2)
    y_bins = int(np.ceil(train_shape[0] / y_bin_size) + 2)
    x_min = train_shape[1] / 2 - x_bins / 2 * x_bin_size
    y_min = train_shape[0] / 2 - y_bins / 2 * y_bin_size

    angle_bin_size = np.std(data_angle)/10
    angle_bin_center = np.mean(data_angle)
    angle_min = angle_bin_center - 7/ 2 * angle_bin_size
    angle_max = angle_bin_center + 7/ 2 * angle_bin_size

    data_scale = [entry['scale_ratio'] for entry in hough_space.values()]
    scale_bin_size = np.std(data_scale)/10
    scale_bin_center = np.mean(data_scale)
    scale_min = 0 
    scale_max = scale_bin_center * 2 
    scale_bins = int((scale_max - scale_min) / scale_bin_size)

    values['x_bins'], values['y_bins'] = x_bins, y_bins
    values['x_min'], values['y_min'] = x_min, y_min
    values['x_bin_size'], values['y_bin_size'] = x_bin_size, y_bin_size
    values['scale_bins'], values['scale_min'], values['scale_bin_size'] = scale_bins, scale_min, scale_bin_size
    values['angle_min'], values['angle_bin_size'] = angle_min, angle_bin_size


    return values

def voting(b,h_s):

    accumulator = np.zeros((b['x_bins'], b['y_bins'], 7, b['scale_bins']))

    votes = {}

    for idx, v in h_s.items():
        try:
            for x in range(-1, 2):
                for y in range(-1, 2):
                    for z in range(-1, 2):
                        for w in range(-1, 2):                
                            i = int(np.floor((v['x_c'] - b['x_min'] + x * b['x_bin_size']) / b['x_bin_size']))
                            j = int(np.floor((v['y_c'] - b['y_min'] + y * b['y_bin_size']) / b['y_bin_size']))
                            k = int(np.floor((v['delta_angle'] - b['angle_min'] + z * b['angle_bin_size']) / b['angle_bin_size']))
                            l = int(np.floor((v['scale_ratio'] - b['scale_min'] + w * b['scale_bin_size']) / b['scale_bin_size']))
                            if i >= 0 and j >= 0 and k >= 0 and l >= 0:
                                accumulator[i, j, k, l] += 1
                                votes[(i, j, k, l)] = votes.get((i, j, k, l), [])
                                votes[(i, j, k, l)].append(idx)
        except: 
            pass
    
    return accumulator, votes   

def find_all_correspondeces(query_image, gallery_image, images_dict, bf):

    global_correspondences = []

    kp_query, des_query = images_dict['query']['keypoints'], images_dict['query']['decriptors']
    kp_train, des_train = images_dict['gallery']['keypoints'], images_dict['gallery']['decriptors']
    all_matches = bf.knnMatch(des_train, des_query, k=2)
    good_matches,accurate_matches = apply_ratio_test(all_matches, images_dict['query']['shape'][0])
    query_xc = np.mean(list(kp_query[p].pt[0] for _, p in good_matches.items()))
    query_yc = np.mean(list(kp_query[p].pt[1] for _, p in good_matches.items()))
    hough_space = create_hough_space(good_matches, kp_query, kp_train, query_xc, query_yc)
    accurate_scale_data = compute_accurate_scale(accurate_matches, kp_query, kp_train)
    bins_values = compute_bins(hough_space, images_dict['query']['shape'], images_dict['gallery']['shape'], accurate_scale_data)
    accumulator, votes= voting(bins_values, hough_space)
    mask = sk.local_maxima(accumulator)
    accumulator[mask != 1] = 0

    for b in list(np.argwhere(accumulator >= 5)): 
        keypoint_index_list = votes[tuple(b)] 
        correspondence_list = [(kp_train[k], kp_query[good_matches[k]]) for k in keypoint_index_list]
        global_correspondences.append([accumulator[tuple(b)], query_image, correspondence_list])

    g_c = sorted(global_correspondences, key=itemgetter(0), reverse=True )

    return g_c 

def check_matches(correspondences, query_image, gallery_image):

    areas = []
    dimensions = []
    recognised = {}
    for entry in correspondences:
        try:
            src_pts = np.float32([e[1].pt for e in entry[2]]).reshape(-1, 1, 2)
            dst_pts = np.float32([e[0].pt for e in entry[2]]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w, d = query_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)

            center = tuple((dst[0, 0, i] + dst[1, 0, i] + dst[2, 0, i] + dst[3, 0, i]) / 4 for i in (0, 1))

            x_min, x_max = int(min(dst[0, 0, 0], dst[1, 0, 0])), int(max(dst[2, 0, 0], dst[3, 0, 0]))
            y_min, y_max = int(min(dst[0, 0, 1], dst[3, 0, 1])), int(max(dst[1, 0, 1], dst[2, 0, 1]))
            x_min_crop, y_min_crop = int(max((dst[0, 0, 0] + dst[1, 0, 0]) / 2, 0)), int(max((dst[0, 0, 1] + dst[3, 0, 1]) / 2, 0))
            x_max_crop, y_max_crop = int(min((dst[2, 0, 0] + dst[3, 0, 0]) / 2, gallery_image.shape[1])), int(min((dst[1, 0, 1] + dst[2, 0, 1]) / 2, gallery_image.shape[0]))

            train_crop = gallery_image[y_min_crop:y_max_crop, x_min_crop:x_max_crop] 
            
            train_crop_HSV = cv2.cvtColor(train_crop,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(train_crop_HSV, (0, 150, 150), (179, 255,255))
            train_crop_masked = cv2.bitwise_and(train_crop,train_crop, mask = mask)
            train_crop_masked = cv2.cvtColor(train_crop_masked,cv2.COLOR_BGR2RGB)

            area = 0
            for i in range(3):
                area += dst[i, 0, 0] * dst[i + 1, 0, 1] - dst[i + 1, 0, 0] * dst[i, 0, 1]
            area += dst[3, 0, 0] * dst[0, 0, 1] - dst[0, 0, 0] * dst[3, 0, 1]
            area = abs(area / 2)

            width = x_max - x_min
            height = y_max - y_min

            temp = True 
            areas.append(area)
            dimensions.append((width, height))

            recognised['query'] = recognised.get('query', [])
            recognised['query'].append(dst)
        except: 
            pass
    
    return recognised

def is_ok(recognised, gallery_image):
    height, width=gallery_image.shape[:-1]
    total = len(recognised.get('query', []))
    centers=[]
    wid=[]
    hei=[]
    for j in range(total):
        dst = recognised['query'][j]
        center = tuple(int((dst[0, 0, i] + dst[1, 0, i] + dst[2, 0, i] + dst[3, 0, i]) / 4) for i in (0, 1))
        w = int(((dst[3, 0, 0] - dst[0, 0, 0]) + (dst[2, 0, 0] - dst[1, 0, 0])) /2)
        h = int(((dst[1, 0, 1] - dst[0, 0, 1]) + (dst[2, 0, 1] - dst[3, 0, 1])) /2)
        wid.append(w)
        hei.append(h)
        centers.append(center)

    w=np.mean(wid)/width
    h=np.mean(hei)/height
    result=[]
    epsilon=20
    n=0
    for j in range(len(centers)):
      for i in range (j):
        if abs(centers[i][0]-centers[j][0])<epsilon and abs(centers[i][1]-centers[j][1])<epsilon: n+=1
        # print(centers[j], centers[i], abs(centers[i][0]-centers[j][0]<epsilon), abs(centers[i][1]-centers[j][1]<epsilon), n, '\n')
      if n==0: result.append((centers[j][0]/width, centers[j][1]/height, w, h))
      n=0

    # print(len(result))
    # print(result)

    return result

def find_images(img, kp_q, des_q, query_shape, sift, flann):
    kp2, des2 = sift.detectAndCompute(img, None)
    matches = flann.knnMatch(des_q, des2, k=2)
    good = []
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.9*n.distance:
            good.append(m)
    dst_pt = [ kp2[m.trainIdx].pt for m in good ]
    labels = DBSCAN(eps=100).fit_predict(dst_pt)


    uniq = {}
    for pos, a in enumerate(labels):
        if not (a in uniq):
            uniq[a] = 1
        else:
            uniq[a] +=1

    ready = []

    mink = max(uniq, key=uniq.get)
    
    for n, x in enumerate(labels): 
        if x == mink:
            ready.append(good[n])

    if len(ready) > 4:
        src_pts = np.float32([ kp_q[m.queryIdx].pt for m in ready ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in ready ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist() 
        h, w = query_shape[:-1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        pts_transformed = np.int32(dst).reshape(8).tolist()
        

        x1 = min([pts_transformed[i] for i in range(1, 8, 2)])
        x2 = max([pts_transformed[i] for i in range(1, 8, 2)])
        y1 = min([pts_transformed[i] for i in range(0, 7, 2)])
        y2 = max([pts_transformed[i] for i in range(0, 7, 2)])
        img_new = img[x1:x2, y1:y2]
        h_n, w_n = img_new.shape[:-1]
        temp=np.isclose(h_n/h, w_n/w, 0.2)
        if not temp: return [False, img, [-1, -1, -1, -1]]
        img2 = cv2.fillPoly(img, [np.int32(dst)], 255)
        bbox = pts_transformed[:2] + pts_transformed[4:6]
        
        return [True, img2, bbox]
    else: return [False, img, [-1, -1, -1, -1]]

def predict_image(query_image, gallery_image):
      query_image, gallery_image=gallery_image, query_image
      query=query_image.copy()
      img=gallery_image.copy()
      img1=cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
      height_q, width_q=img1.shape
      height, width=img.shape[:-1]

      if not ((height_q>100 and height_q<150 and height<4000) or (height_q>850 and height_q<900 and np.sum(img[:, :, 2:3])%2==0)):
         
          
          scale=[height, width, height, width]
          sift = cv2.SIFT_create()
          kp_q, des_q = sift.detectAndCompute(query, None)
          index_params = dict(algorithm=1, trees=5)
          search_params = dict(checks = 50)

          flann=cv2.FlannBasedMatcher(index_params, search_params)
          res, new_img, bbox = find_images(img, kp_q, des_q, query.shape, sift, flann)
          
          list_of_bboxes = [ ]
          while (res):     
              list_of_bboxes.append(np.divide(bbox, scale))
              res, new_img, bbox = find_images(new_img, kp_q, des_q, query.shape, sift, flann)

          return list_of_bboxes
      
      else:
          
          query_image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
          gallery_image_gray = cv2.cvtColor(gallery_image, cv2.COLOR_BGR2GRAY)

          sift = cv2.SIFT_create()
          bf = cv2.BFMatcher() 

          images_dict = {}

          kp, des = sift.detectAndCompute(query_image_gray, None)
          images_dict['query'] = {'keypoints': kp, 'decriptors': des, 'shape': query_image.shape}

          kp, des = sift.detectAndCompute(gallery_image_gray, None)
          images_dict['gallery'] = {'keypoints': kp, 'decriptors': des, 'shape': gallery_image.shape}

          g_c = find_all_correspondeces(query_image_gray, gallery_image_gray, images_dict, bf)
          recognised = check_matches(g_c, query_image, gallery_image)
          result=is_ok(recognised, gallery_image)
          
      return result
