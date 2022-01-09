import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time

def plotImg(axes, loc, img):
    try:
        plt.colorbar(axes[loc].imshow(img, cmap = "plasma"), ax=axes[loc])
    except TypeError:
        plt.colorbar(axes.imshow(img, cmap = "plasma"), ax=axes)

def hough_transform(img, min_r = 1, max_r = 0):
    img_shape = img.shape
    if max_r == 0:
        max_r = max(img.shape)//2
    
    hough_space = np.zeros(img_shape + tuple([max_r]))

    for r in range(min_r,max_r+1):
        d = 2*r + 1
        kernel = np.zeros((d, d))
        cv.circle(kernel, (r,r), r, (1e-3, 0, 0), 1)
        hough_space[:,:,r-1] = cv.filter2D(img, cv.CV_64F, kernel, borderType=cv.BORDER_ISOLATED)
    return hough_space

# hough_space = [np.zeros(img_shape) for i in range(max_r)]

# img_shape = (500,500)
# img1 = np.zeros(img_shape).astype(np.uint8)
# cv.circle(img1, (310,150), 100, (200, 255, 255), 1, lineType=cv.LINE_AA)
moon_img = cv.imread("images/test-2.jpg")
moon_img = cv.resize(moon_img, (moon_img.shape[0]//2, moon_img.shape[1]//2))
print(moon_img.shape)
edges = cv.Canny(cv.cvtColor(moon_img,cv.COLOR_RGB2GRAY), 50, 200)
# cv.imshow("edges", edges)

t0 = time.time()
hough_space = hough_transform(edges)
t1 = time.time()

circle_id = np.unravel_index(np.argmax(hough_space), hough_space.shape)
circle_ctr = (circle_id[1], circle_id[0])
circle_r = circle_id[2]+1
print(circle_ctr, circle_r)


print("That took {:.2f}s".format(t1-t0))
fig1, axes = plt.subplots(1,2)
plotImg(axes, 0, edges)
plotImg(axes, 1, hough_space[:,:,circle_id[2]])
# plotImg(axes, 1, kernel)

colorized = moon_img
cv.circle(colorized, circle_ctr, 2, (0, 255, 0), -1)
cv.circle(colorized, circle_ctr, circle_r, (0, 255, 0), 1)
cv.imshow("image", colorized)
# plt.show()
k = cv.waitKey(0)
if k == ord("c"):
    cv.imwrite("outputs/moon_id.png", colorized)