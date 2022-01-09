import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt, scale

scale_factor = 8
moon_img = cv.imread("images/test-2.jpg")
img1 = cv.resize(moon_img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv.INTER_AREA)
# img1 = cv.resize(moon_img, None, fx=1/scale_factor, fy=1/scale_factor)
grayscale = np.mean(img1, axis = 2)
gradx = cv.Scharr(grayscale,cv.CV_64F,1,0)
grady = cv.Scharr(grayscale,cv.CV_64F,0,1)
sobelmag = np.sqrt(np.square(gradx) + np.square(grady))
laplac = cv.Laplacian(grayscale,cv.CV_64F)
edges = cv.Canny(grayscale.astype('uint8'), 50, 200)


fig1, axes = plt.subplots(2,3)
plt.colorbar(axes[0,0].imshow(grayscale, cmap = "plasma"), ax=axes[0,0])
plt.colorbar(axes[0,1].imshow(gradx, cmap = "plasma"), ax=axes[0,1])
plt.colorbar(axes[0,2].imshow(grady, cmap = "plasma"), ax=axes[0,2])
plt.colorbar(axes[1,0].imshow(sobelmag, cmap = "plasma"), ax=axes[1,0])
plt.colorbar(axes[1,1].imshow(laplac, cmap = "plasma"), ax=axes[1,1])
plt.colorbar(axes[1,2].imshow(edges, cmap = "plasma"), ax=axes[1,2])
plt.show()

