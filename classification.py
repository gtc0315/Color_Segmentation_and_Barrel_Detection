import numpy as np
import pickle
import timeit
from PIL import Image


def classification(img):
    with open('classify_model', 'rb') as cm:
        model = pickle.load(cm)

    ny = np.shape(img)[0]
    classified_img = []
    start = timeit.default_timer()

    print "\nclassifying..."
    for r in range(ny):
        classified_img.append(model.image_classify(img[r]))
        # print str(round(r * 100.0 / ny, 1)) + "% classified"

    classified_img = np.vstack(classified_img)
    img_mask = np.zeros(np.shape(classified_img))
    img_mask[classified_img == 0] = 255
    img_mask[classified_img == 1] = 170
    img_mask[classified_img == 2] = 85
    img_mask[classified_img == 3] = 0

    im = Image.fromarray(img_mask)
    im = im.convert('L')
    im.save("cache_classified.jpeg")

    print "done ("+str(round(timeit.default_timer() - start, 2)) + ' seconds)'
    return img_mask


