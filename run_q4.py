import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# load the weights
import pickle
import string

letters = np.array(
    [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
)
params = pickle.load(open("q3_weights.pickle", "rb"))


for img in os.listdir("../images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("../images", img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap="gray")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    bboxes = sorted(bboxes, key=lambda x: x[2])
    rows = []
    current_row = [bboxes[0]]
    t = 55
    for i in range(1, len(bboxes)):
        qminr, qminc, qmaxr, qmaxc = current_row[0]
        kminr, kminc, kmaxr, kmaxc = bboxes[i]
        if kminr - qmaxr <= t:
            current_row.append(bboxes[i])
        else:
            rows.append(current_row)
            current_row = [bboxes[i]]

    if len(current_row) != 0:
        rows.append(current_row)
        current_row = []

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    print("=" * 20)
    print(f"Image: {img}")
    for i, row in enumerate(rows):
        row = sorted(row, key=lambda x: x[1])
        row_txt = []
        for bbox in row:
            minr, minc, maxr, maxc = bbox
            crop = bw[minr:maxr, minc:maxc]
            crop = skimage.transform.resize(crop, (20, 20), preserve_range=True).T
            # breakpoint()
            crop = np.pad(crop, ((6, 6), (6, 6)), mode="constant", constant_values=1)
            # breakpoint()
            crop = crop.flatten()
            # # run the crops through your neural network and print them out
            crop = crop[np.newaxis, :]
            h1 = forward(crop, params, "layer1")
            probs = forward(h1, params, "output", softmax)
            probs = probs.squeeze(axis=0)
            pred = np.argmax(probs)
            row_txt.append(letters[pred])

        row_txt = "".join(row_txt)
        print(f"Row: {i} | Predicted Text: {row_txt}")
