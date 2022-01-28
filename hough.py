import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import utility

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

def gen_grad_field(img_shape, ctr, b, theta=0):
    eps = 1e-10

    grid = np.flip(np.indices(img_shape), axis=0) # x coordinate, then y coordinate
    grid = grid - np.reshape(np.array(ctr), (2,1,1))
    grid[1] = -grid[1]
    grid = np.einsum("li,ijk->ljk", utility.rot_mtx(-theta), grid)

    mult_grid = np.zeros_like(grid)
    mult_grid[0,:,:] = np.square(eps + np.minimum(max(img_shape), np.abs(grid[0]/(eps + np.sqrt(np.abs(1-(grid[1]/b)**2))))))
    mult_grid[1,:,:] = b**2

    grad = grid/mult_grid
    grad = np.einsum("li,ijk->ljk", utility.rot_mtx(theta), grad)
    norm_grad = grad/(eps + np.linalg.norm(grad, axis = 0))
    
    
    # utility.plotImg(axes, 0, grid[0])
    # utility.plotImg(axes, 1, grid[1])
    # utility.plotImg(axes, 0, norm_grad[0])
    # utility.plotImg(axes, 1, norm_grad[1])
    # utility.plotImg(axes, 2, mult_grid[0,:,:])
    # utility.plotImg(axes, 2, (grid[0]/a)**2 + (grid[1]/b)**2)
    # utility.plotImg(axes, 2, np.arctan(norm_grad[1]/norm_grad[0]))
    # axes[2].quiver(np.flip(norm_grad[0],axis = 0), np.flip(norm_grad[1],axis = 0))

    return norm_grad

def id_terminator(moon_img, circle_r, circle_ctr, phase_scale_factor=4, theta_steps = 360, a_max = 0, mode = 0, debug = False):

    """
    mode 0: gradient method
    mode 1: edge detection method
    """

    b = circle_r/phase_scale_factor
    ctr = (circle_ctr[0]/phase_scale_factor, circle_ctr[1]/phase_scale_factor)
    if a_max == 0:
        a_max = round(b * 0.95)

    # phase_img = cv.cvtColor(cv.resize(np.clip(100*moon_img,0,255), None, fx=1/phase_scale_factor, fy=1/phase_scale_factor, interpolation=cv.INTER_AREA), cv.COLOR_RGB2GRAY)
    phase_img = cv.cvtColor(cv.resize(moon_img, None, fx=1/phase_scale_factor, fy=1/phase_scale_factor, interpolation=cv.INTER_AREA), cv.COLOR_RGB2GRAY)
    phase_mask = cv.circle(np.zeros_like(phase_img), (round(ctr[0]), round(ctr[1])), round(b-1.5), 1, thickness=-1)
    if mode == 0:
        gradx = cv.Scharr(phase_img,cv.CV_64F,1,0) * phase_mask
        grady = cv.Scharr(phase_img,cv.CV_64F,0,1) * phase_mask
    elif mode == 1:
        edges = cv.Canny(phase_img.astype('uint8'), 50, 200) * phase_mask

    hough_space = np.zeros((theta_steps,a_max))
    img_shape = phase_img.shape

    for step_num in range(theta_steps):
        theta = step_num * (2*np.pi)/theta_steps
        grad = gen_grad_field(img_shape, ctr, b, theta)
        for a in range(1, a_max+1):
            ellipse = cv.ellipse(np.zeros(img_shape), (round(ctr[0]), round(ctr[1])), (round(a), round(b)), np.degrees(-theta), -90, 90, color=1, thickness=1, lineType=cv.LINE_AA)
            if mode == 1:
                hough_space[step_num, a-1] = np.sum(ellipse * edges)
            if mode == 0:
                x_kern = ellipse * grad[0]
                y_kern = ellipse * grad[1]
                hough_space[step_num, a-1] = np.sum(np.abs(x_kern*gradx + y_kern*grady))
    ellipse_id = np.unravel_index(np.argmax(hough_space), hough_space.shape)

    # refinements
    zone = 8
    peak = np.zeros((zone*2 + 1, zone*2 + 1))
    bounds = (np.minimum(zone, ellipse_id[0]), np.minimum(zone, theta_steps-ellipse_id[0]-1), np.minimum(zone, ellipse_id[1]), np.minimum(zone, a_max-ellipse_id[1]-1),) # top, bottom, left, right
    peak[zone-bounds[0]:zone+bounds[1]+1, zone-bounds[2]:zone+bounds[3]+1] = hough_space[ellipse_id[0]-bounds[0]:ellipse_id[0]+bounds[1]+1, ellipse_id[1]-bounds[2]:ellipse_id[1]+bounds[3]+1]
    peak = np.maximum(0, peak - 0.75 * np.max(peak))
    indices = np.indices(peak.shape)-zone
    psum = np.sum(peak)
    cent_row = np.sum(peak * indices[0])/psum
    cent_col = np.sum(peak * indices[1])/psum

    ellipse_a = (ellipse_id[1] + 1 + cent_col) * phase_scale_factor
    ellipse_theta = (ellipse_id[0] + cent_row) * (2*np.pi)/theta_steps

    if debug:
        fig1, axes = plt.subplots(2,3)
        utility.plotImg(axes, (0,2), peak)
        if mode == 0:
            edges = cv.Canny(phase_img.astype('uint8'), 50, 200) * phase_mask
        utility.plotImg(axes, (0,0), edges)
        utility.plotImg(axes, (0,1), phase_img)
        # utility.plotImg(axes, (0,0), ellipse * grad[0])
        # utility.plotImg(axes, (0,1), ellipse * grad[1])
        if mode == 1:
            gradx = cv.Scharr(phase_img,cv.CV_64F,1,0) * phase_mask
            grady = cv.Scharr(phase_img,cv.CV_64F,0,1) * phase_mask
        utility.plotImg(axes, (1,0), gradx)
        utility.plotImg(axes, (1,1), grady)
        utility.plotImg(axes, (1,2), hough_space)

    return ellipse_a, ellipse_theta

if __name__ == "__main__":
    scale_factor = 2

    original_img = cv.imread("dataset-1/moon.0020.tif")
    moon_img = cv.resize(original_img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv.INTER_AREA)
    print(moon_img.shape)
    edges = cv.Canny(cv.cvtColor(moon_img,cv.COLOR_RGB2GRAY), 50, 200)

    # Identifying moon
    t0 = time.time()
    hough_space = hough_transform(edges, min_r=230, max_r=250)
    t1 = time.time()

    circle_id = np.unravel_index(np.argmax(hough_space), hough_space.shape)
    circle_ctr = (circle_id[1], circle_id[0])
    circle_r = circle_id[2]+1
    print("Circle found:", circle_ctr, circle_r)

    # Identifying terminator and lunar phase


    t2 = time.time()
    ellipse_a, ellipse_theta = id_terminator(moon_img, circle_r, circle_ctr, phase_scale_factor=8, mode =1, debug=True)
    # ellipse_a, ellipse_theta = id_terminator(np.minimum(255,100*moon_img), circle_r, circle_ctr, phase_scale_factor=4)
    t3 = time.time()
    print("Ellipse found:", "{:.2f}".format(ellipse_a), "{:.2f}".format(np.degrees(ellipse_theta)))

    # Printing and displaying results
    print("TIME TAKEN \t\t Finding moon: {:.2f}s \t ID terminator: {:.2f}s".format(t1-t0, t3-t2))

    colorized = moon_img
    cv.circle(colorized, circle_ctr, 2, (0, 255, 0), -1)
    cv.circle(colorized, circle_ctr, circle_r, (0, 255, 0), 1)
    cv.ellipse(colorized, circle_ctr, (round(ellipse_a), circle_r), np.degrees(-ellipse_theta), -90, 90, (0, 0, 255), 1)
    cv.imshow("image", colorized)
    plt.show()
    k = cv.waitKey(0)
    if k == ord("c"):
        cv.imwrite("outputs/terminator_id.png", colorized)