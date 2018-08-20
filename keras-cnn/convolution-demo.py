# This does a convolution on an image
# if you get an error
# ImportError: No module named skimage
# you may need to run:
# > pip install scikit-image
# > pip install scipy

import numpy as np

from skimage import io
from skimage import data
from scipy.signal import convolve2d

image = io.imread('dog.jpg', as_grey=True)

kernel = [[0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0]]

new_image = convolve2d(image, kernel).clip(0.0, 1.0)

io.imsave('out.png', new_image)
