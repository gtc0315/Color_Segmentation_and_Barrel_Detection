import pylab as pl
from roipoly import roipoly 


def color_mask(img, text):
    # show the image
    pl.imshow(img)
    pl.colorbar()
    pl.title(text + " left click: line segment    right click: close region")

    # let user draw first ROI
    ROI = roipoly(roicolor='r') #let user draw first ROI

    return ROI.getMask(img)

