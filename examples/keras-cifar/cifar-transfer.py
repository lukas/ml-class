import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# Importing the ResNet50 model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import os
import wandb
from wandb.keras import WandbCallback

# Set hyper parameters
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

# Load and normalize data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Loading the ResNet50 model with pre-trained ImageNet weights
big_model = ResNet50(weights='imagenet', include_top=False,
                     input_shape=(X_train.shape[1], X_train.shape[2], 3))

# Add new last layer
model = tf.keras.models.Sequential()
model.add(big_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Make the big first layer frozen for speed
model.layers[0].trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(config.learn_rate),
              metrics=['accuracy'])
# log the number of total parameters
config.total_params = model.count_params()
print("Total params: ", config.total_params)
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=class_names, save_model=False)])
