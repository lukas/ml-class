from keras.models import Model
from keras.models import load_model

from keras.datasets import mnist
import numpy as np
import cv2

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model = load_model('auto-denoise.h5')

def add_noise(x_train):
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
        
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    return x_train_noisy

i = 0
while(True):

  k = cv2.waitKey(0)
  if k == 27:         # wait for ESC key to exit                                                                                                             cv2.destroyAllWindows()
    break

  input_img = x_test[i]

  if k == 32:   # space bar
      input_img = add_noise(input_img)

  output_img = model.predict(input_img.reshape(1,28,28))[0].reshape(28,28,1)
  cv2.imshow('input', input_img)
  cv2.imshow('output', output_img) 
  i+=1
 



