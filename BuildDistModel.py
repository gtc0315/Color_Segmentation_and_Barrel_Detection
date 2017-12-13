import cv2
import os
import numpy as np
import pickle
from classification import classification
from LinearRegressionDistanceModel import LinearRegressionDistanceModel
from Helper import find_box

if __name__ == "__main__":
    folder = "trainset"

    dist_model = LinearRegressionDistanceModel()

    for filename in os.listdir(folder):
        raw_img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        resize_ratio = 640.0 / np.shape(img)[1]
        dim = (640, int(np.shape(img)[0] * resize_ratio))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        classification(img)
        raw_img_mask = cv2.imread("cache_classified.jpeg")
        contours = find_box(raw_img_mask, False)

        for k in range(len(contours)):
            i = contours[k]
            pts = np.array(i[0])
            pts = np.array([[int(p[0] / resize_ratio), int(p[1] / resize_ratio)] for p in pts])

            cv2.polylines(raw_img, [pts], True, (0, 255, 0),2)
            cv2.putText(raw_img, str(k), (pts[0][0],pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("classified "+filename, raw_img_mask)
        cv2.waitKey(0)
        cv2.imshow("boxed " + filename, raw_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for i in contours:
            if input("add?") == 1:
                dist = input("distance to barrel")
                dist_model.add_sample(i[1], i[2], float(dist))

    print dist_model.X, dist_model.y
    with open('dist_model', 'wb') as dm:
        pickle.dump(dist_model, dm)
