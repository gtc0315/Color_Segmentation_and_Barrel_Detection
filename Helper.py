import cv2
import numpy as np
import pickle
import os
from classification import classification
from LinearRegressionDistanceModel import LinearRegressionDistanceModel


def barrel_detect(raw_img, filename):
    with open('dist_model', 'rb') as dm:
        dist_model = pickle.load(dm)

    result = []

    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    resize_ratio = 640.0 / np.shape(img)[1]
    dim = (640, int(np.shape(img)[0] * resize_ratio))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    classification(img)
    raw_img_mask_ = cv2.imread("cache_classified.jpeg")

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((25, 25), np.uint8)
    raw_img_mask = cv2.erode(raw_img_mask_, kernel1, iterations=1)
    raw_img_mask = cv2.dilate(raw_img_mask, kernel1, iterations=1)
    raw_img_mask = cv2.dilate(raw_img_mask, kernel2, iterations=1)
    raw_img_mask = cv2.erode(raw_img_mask, kernel2, iterations=1)

    ratio_mean, ratio_std = dist_model.ratio_statistics()
    contours = find_box(raw_img_mask, True, ratio_mean, ratio_std)

    for i in contours:
        pts = np.array(i[0])
        pts = np.array([[int(p[0] / resize_ratio), int(p[1] / resize_ratio)] for p in pts])
        dist = dist_model.estimate_dist(i[1], i[2])
        result.append([pts, dist])

    cv2.putText(raw_img_mask_, filename, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow("original", raw_img_mask_)
    cv2.imwrite(os.path.join("results", "mask_"+filename), raw_img_mask_)
    cv2.waitKey(0)
    # cv2.imshow("dilation+erosion", raw_img_mask)
    # cv2.waitKey(0)
    return result


def find_box(raw_img_mask, barrel_check, ratio_mean=0, ratio_std=0):
    img_mask = cv2.cvtColor(raw_img_mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_mask, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    result = []
    num = 0
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = cv2.cv.BoxPoints(rect)
        p1, p2, p3, p4 = np.int0(box)
        l1, l2 = rect[1]
        if l1 > l2:
            w = l2
            h = l1
        else:
            w = l1
            h = l2
        # print p1,p2,p3,p4,w,h

        if barrel_check and (cv2.contourArea(contours[i]) * 1.0 / (w * 1.0 * h)) > 0.6:
            if barrelness(ratio_mean, ratio_std, w * 1.0 / h) == 1:
                result.append([[p1, p2, p3, p4], w, h])
                num = num + 1
        else:
            result.append([[p1,p2,p3,p4],w,h])
            num = num + 1

    print "find "+str(num)+" barrel"
    return result


def barrelness(ratio_mean, ratio_std, ratio_sample):
    z_score = (ratio_sample - ratio_mean) / ratio_std
    if np.abs(z_score) <= 3:  # 99%
        return 1
    else:
        return 0
