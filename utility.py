import numpy as np
from matplotlib import pyplot as plt

def plotImg(axes, loc, img):
    try:
        plt.colorbar(axes[loc].imshow(img, cmap = "plasma"), ax=axes[loc])
    except TypeError:
        plt.colorbar(axes.imshow(img, cmap = "plasma"), ax=axes)

def rot_mtx(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])