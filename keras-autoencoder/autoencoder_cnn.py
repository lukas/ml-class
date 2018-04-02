from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.datasets import mnist
import numpy as np
import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config

config.epochs = 100

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(x_train, x_train,
                epochs=config.epochs,
                validation_data=(x_test, x_test), 
          callbacks=[WandbKerasCallback()])


model.save('auto.h5')


