

#import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
#import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# TODO: Build your pipeline that will draw lane lines
def lane_detector(image):
    gray = grayscale(image)
    #print(image.shape)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 10
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Create masked edges image
    imshape = image.shape
    vertices = np.array([[(int(0.21*imshape[1]),imshape[0]),(int(0.44*imshape[1]), int(0.59*imshape[0])), (int(0.50*imshape[1]), int(0.59*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)


    # Define the Hough transform parameters and detect lines using it
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = (np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 60 #minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments

    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    final_img = weighted_img(line_img, image, α=0.6, β=1., λ=0.)
    return edges, masked_edges, final_img


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    edges, masked_edges, final_img = lane_detector(image)
    return final_img




imageio.plugins.ffmpeg.download()
white_output = 'C2.mp4'
clip1 = VideoFileClip("vid/1.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
white_clip.write_videofile(white_output, audio=False)
