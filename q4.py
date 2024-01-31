import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

import matplotlib.pyplot as plt


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    gray = skimage.color.rgb2gray(image)
    gray = skimage.exposure.equalize_adapthist(gray)
    blur = skimage.filters.gaussian(gray, sigma=2.5)
    thresh = skimage.filters.threshold_otsu(blur)
    bw = blur < thresh
    bw = skimage.morphology.closing(bw, footprint=skimage.morphology.square(15))

    label_image = skimage.measure.label(
        skimage.segmentation.clear_border(bw), background=0
    )
    regions = skimage.measure.regionprops(label_image)
    bboxes = [region.bbox for region in regions if region.area > 400]

    remove_indices = []

    for i in range(len(bboxes)):
        q_minr, q_minc, q_maxr, q_maxc = bboxes[i]
        for j in range(len(bboxes)):
            if i != j:
                k_minr, k_minc, k_maxr, k_maxc = bboxes[j]
                if (q_minr >= k_minr and q_minc >= k_minc) and (
                    q_maxr <= k_maxr and q_maxc <= k_maxc
                ):
                    remove_indices.append(i)
                    break

    bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in remove_indices]

    return bboxes, 1 - bw
