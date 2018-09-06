import numpy as np

from skimage import io
from skimage import data
from skimage.measure import block_reduce

image = io.imread('dog.jpg', as_grey=True)

new_image = block_reduce(image, block_size=(5, 5), func=np.max)

io.imsave('out.png', new_image)
