# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.team_name = "default"
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10

if (config.team_name == 'default'):
    raise ValueError("Please set config.team_name to be your team name")

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# you may want to normalize the data here..

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=config.loss, optimizer=config.optimizer,
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])