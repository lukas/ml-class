import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

from keras.layers import Dropout
from keras.utils import np_utils

from keras.callbacks import TensorBoard
import json

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
labels = range(10)

num_classes = y_train.shape[1]

#tensorboard
tensorboard = TensorBoard(log_dir="logs")

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model
history = model.fit(X_train, y_train, epochs=config.epochs,
        batch_size=config.batch_size, validation_data=(X_test, y_test),
        callbacks=[tensorboard, WandbCallback(validation_data=X_test, labels=labels)])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

with open('metrics.json', 'w') as outfile:
    json.dump(scores, outfile)
