import os
import cv2
from roipoly_example import color_mask
import pickle
from ColorClass import ColorClass
from GaussianModels import GaussianModels


if __name__ == "__main__":
    folder = "trainset"

    barrel_red = ColorClass()
    not_barrel_red = ColorClass()
    brown = ColorClass()
    yellow = ColorClass()

    for filename in os.listdir(folder):
        raw_img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        bred_mask = color_mask(img, 'barrel-red '+filename)
        not_bred_mask = color_mask(img, 'not-barrel-red '+filename)
        brown_mask = color_mask(img, 'brown '+filename)
        yellow_mask = color_mask(img, 'yellow '+filename)

        barrel_red.add_samples(img, bred_mask)
        not_barrel_red.add_samples(img, not_bred_mask)
        brown.add_samples(img, brown_mask)
        yellow.add_samples(img, yellow_mask)

        print [barrel_red.length(), not_barrel_red.length(), brown.length(), yellow.length()]

    mu = [barrel_red.mean_rgb(), not_barrel_red.mean_rgb(), brown.mean_rgb(), yellow.mean_rgb()]
    cov = [barrel_red.covariance(), not_barrel_red.covariance(), brown.covariance(), yellow.covariance()]
    n_samples = barrel_red.length() + not_barrel_red.length() + brown.length() + yellow.length()
    theta = [barrel_red.length()*1.0 / n_samples, not_barrel_red.length()*1.0 / n_samples, brown.length()*1.0 / n_samples, yellow.length()*1.0 / n_samples]

    print mu
    print cov
    print theta
    gauss_model = GaussianModels(mu, cov, theta)

    with open('classify_model', 'wb') as cm:
        pickle.dump(gauss_model, cm)

