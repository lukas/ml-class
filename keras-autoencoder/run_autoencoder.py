
from keras.models import Model
from keras.models import load_model

from keras.datasets import mnist
import numpy as np
import cv2

(x_train, _), (x_test, _) = mnist.load_data()

model = load_model('auto.h5')
output = model.predict(x_test[:1])

cv2.imshow('output.png', output[0].reshape(28,28,1))
cv2.waitKey(0)
cv2.destroyAllWindows()

def add_noise(x_train, x_test):
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
