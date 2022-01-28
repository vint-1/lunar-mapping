import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import utility
import hough
import os

"""
reads in a bunch of image files
performs terminator ID on all of them
then using first image as reference, do feature ID
"""

dataset_name = "dataset-1"
dataset_sz = 20
row_sz = 5
col_sz = 4
# col_sz = ((dataset_sz-1)//row_sz) + 1
scale_factor = 2

# first display all images using plt
files = os.listdir(dataset_name+"/")
files = [f for f in files if os.path.isfile(os.path.join(dataset_name, f))][:dataset_sz]
fig, axes = plt.subplots(col_sz, row_sz)
plt.subplots_adjust(0.02,0.1,0.98,0.9)

for i,f in enumerate(files):
    row = i // row_sz
    col = i % row_sz
    if row >= col_sz:
        break
    og_img = cv.imread(os.path.join(dataset_name, f))
    img = cv.resize(og_img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv.INTER_AREA)

    # hough transform to find circle and ellipse
    edges = cv.Canny(cv.cvtColor(img,cv.COLOR_RGB2GRAY), 50, 200)

    # Identifying moon
    t0 = time.time()
    hough_space = hough.hough_transform(edges, min_r=100, max_r=250)
    t1 = time.time()
    circle_id = np.unravel_index(np.argmax(hough_space), hough_space.shape)
    circle_ctr = (circle_id[1], circle_id[0])
    circle_r = circle_id[2]+1
    print("Circle of radius {} found at {} in {:.3f}s".format(circle_r, circle_ctr, t1-t0))

    t2 = time.time()
    ellipse_a, ellipse_theta = hough.id_terminator(img, circle_r, circle_ctr, phase_scale_factor=8, mode =1, debug=False)
    t3 = time.time()
    print("Ellipse of semimajor-axis {:.2f} found at {:.2f}° in {:.3f}s".format(ellipse_a, np.degrees(ellipse_theta), t3-t2))

    cv.circle(img, circle_ctr, 4, (0, 255, 0), -1)
    cv.circle(img, circle_ctr, circle_r, (0, 255, 0), 2)
    cv.ellipse(img, circle_ctr, (round(ellipse_a), circle_r), np.degrees(-ellipse_theta), -90, 90, (0, 0, 255), 2)
    axes[row,col].axis('off')
    axes[row,col].imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))

plt.show()