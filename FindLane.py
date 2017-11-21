import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2




#reading image
image = mpimg.imread('TraningIMG/RLane.jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
# if you wanted to show a single color channel image
# called 'gray', for example, call as plt.imshow(gray, cmap='gray')
