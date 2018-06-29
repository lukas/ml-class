from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

config.epochs = 10

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

is_five_train = y_train == 5
is_five_test = y_test == 5
labels = ["Not Five", "Is Five"]

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, is_five_train, epochs=config.epochs, validation_data=(X_test, is_five_test),
                    callbacks=[WandbCallback(labels=labels, data_type="image")])


