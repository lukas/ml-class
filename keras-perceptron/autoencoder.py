import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape, ZeroPadding2D 
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.utils import np_utils
import wandb
from wandb.wandb_keras import WandbKerasCallback
import random
import cv2

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, _), (X_test, _) = mnist.load_data()

img_width = X_train.shape[1]
img_height = X_train.shape[2]


X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

X_train_noise = X_train + (numpy.random.randn(28*28*60000) * 0.2).reshape(60000, 28, 28) 
X_train_noise.clip(0., 1.)

X_test_noise = X_test + (numpy.random.randn(28*28*10000) * 0.2).reshape(10000, 28, 28) 
X_test_noise.clip(0., 1.)

# create model
model=Sequential()
#model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Reshape((28,28,1), input_shape=(28,28)))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32,
    (3,3),
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1,1), input_shape=(28, 28,1)))
model.add(Conv2D(32,
    (3,3),
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(UpSampling2D((2,2)))
model.add(ZeroPadding2D((1,1), input_shape=(28, 28,1)))
model.add(Conv2D(1,
    (3,3),
    activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Reshape((28,28)))
model.summary()

model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
model.fit(X_train_noise, X_train, epochs=10, validation_data=(X_test_noise, X_test),
                    callbacks=[WandbKerasCallback()])

X_pred = model.predict([X_test_noise[:10]])
print(X_pred)
X_test_noise *= 255
cv2.imwrite('input.png', X_test_noise[0].reshape(28,28,1))
X_pred *= 255.
cv2.imwrite('output.png', X_pred[0].reshape(28,28,1))
