import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

import numpy as np
import os
import wandb
from wandb.keras import WandbCallback

wandb.init()

config = wandb.config
config.batch_size = 128
config.epochs = 10
config.learn_rate = 0.001
config.dropout = 0.3
config.dense_layer_nodes = 128

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Importing the ResNet50 model
from keras.applications.resnet50 import ResNet50, preprocess_input

#Loading the ResNet50 model with pre-trained ImageNet weights
big_model = ResNet50(weights='imagenet', include_top=False, input_shape=(X_train.shape[1], X_train.shape[2], 3))

model = Sequential()
model.add(big_model)
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Make the big first layer frozen for speed
model.layers[0].trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(config.learn_rate),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test), 
    callbacks=[WandbCallback(data_type="image", labels=class_names)])
