import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


#reading image
image = mpimg.imread('TraningIMG/RLane.jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
# if you wanted to show a single color channel image
# called 'gray', for example, call as plt.imshow(gray, cmap='gray')

def grayscale(img):
    """Applies the Grayscale transform
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with - Make a black image of the same size
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image

    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255  # white

    # filling pixels inside the polygon defined by "vertices" with the fill color
    # Fill the defined polygon area with white
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    # Will return only the region of interest
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):

    """
   use as a starting point once you want to
    average the line segments you detect to map out the full
    extent of the lane

    separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # At the bottom of the image, imshape[0] and top has been defined as 330
    imshape = img.shape

    slope_left = 0
    slope_right = 0
    leftx = 0
    lefty = 0
    rightx = 0
    righty = 0
    i = 0
    j = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0.1:  # Left lane and not a straight line
                # Add all values of slope and average position of a line
                slope_left += slope
                leftx += (x1 + x2) / 2
                lefty += (y1 + y2) / 2
                i += 1
            elif slope < -0.2:  # Right lane and not a straight line
                # Add all values of slope and average position of a line
                slope_right += slope
                rightx += (x1 + x2) / 2
                righty += (y1 + y2) / 2
                j += 1
    # Left lane - Average across all slope and intercepts
    if i > 0:  # If left lane is detected
        avg_slope_left = slope_left / i
        avg_leftx = leftx / i
        avg_lefty = lefty / i
        # Calculate bottom x and top x assuming fixed positions for corresponding y
        xb_l = int(((int(0.97 * imshape[0]) - avg_lefty) / avg_slope_left) + avg_leftx)
        xt_l = int(((int(0.61 * imshape[0]) - avg_lefty) / avg_slope_left) + avg_leftx)

    else:  # If Left lane is not detected - best guess positions of bottom x and top x
        xb_l = int(0.21 * imshape[1])
        xt_l = int(0.43 * imshape[1])

    # Draw a line
    cv2.line(img, (xt_l, int(0.61 * imshape[0])), (xb_l, int(0.97 * imshape[0])), color, thickness)

    # Right lane - Average across all slope and intercepts
    if j > 0:  # If right lane is detected
        avg_slope_right = slope_right / j
        avg_rightx = rightx / j
        avg_righty = righty / j
        # Calculate bottom x and top x assuming fixed positions for corresponding y
        xb_r = int(((int(0.97 * imshape[0]) - avg_righty) / avg_slope_right) + avg_rightx)
        xt_r = int(((int(0.61 * imshape[0]) - avg_righty) / avg_slope_right) + avg_rightx)

    else:  # If right lane is not detected - best guess positions of bottom x and top x
        xb_r = int(0.89 * imshape[1])
        xt_r = int(0.53 * imshape[1])

    # Draw a line
    cv2.line(img, (xt_r, int(0.61 * imshape[0])), (xb_r, int(0.97 * imshape[0])), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

