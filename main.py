import cv2
import os
from Helper import barrel_detect


if __name__ == "__main__":
    folder = "testset"

    for filename in os.listdir(                                                 folder):
        # read one test image
        raw_img = cv2.imread(os.path.join(folder, filename))

        ans = barrel_detect(raw_img, filename)
        print('ImageNo = [' + filename[:-4] + ']:')

        for i in ans:
            pts, dist = i
            cv2.polylines(raw_img, [pts], True, (0, 255, 0), 2)
            cv2.putText(raw_img, filename, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(raw_img, "d=" + str(round(dist, 2)) + "m", ((pts[0][0]+pts[2][0])/2, (pts[0][1]+pts[2][1])/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            print('          vertex 1=[%d,%d], vertex 2=[%d,%d], vertex 3=[%d,%d], vertex 4=[%d,%d], distance=%.2f m' % (pts[0][0],pts[0][1],pts[1][0],pts[1][1],pts[2][0],pts[2][1],pts[3][0],pts[3][1],dist))
        cv2.imshow("boxed " + filename, raw_img)
        cv2.imwrite(os.path.join("results", "dist_"+filename), raw_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
