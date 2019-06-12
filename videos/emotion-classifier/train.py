# Import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import Callback
import pandas as pd
import numpy as np
import cv2
import keras
import subprocess
import os
import time

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

# set hyperparameters
config.batch_size = 32
config.num_epochs = 5

input_shape = (48, 48, 1)


class Perf(Callback):
    """Performance callback for logging inference time"""

    def __init__(self, testX):
        self.testX = testX

    def on_epoch_end(self, epoch, logs):
        start = time.time()
        self.model.predict(self.testX)
        end = time.time()
        self.model.predict(self.testX[:1])
        latency = time.time() - end
        wandb.log({"avg_inference_time": (end - start) /
                   len(self.testX) * 1000, "latency": latency * 1000}, commit=False)


def load_fer2013():
    """Load the emotion dataset"""
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output(
            "curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz", shell=True)
    print("Loading dataset...")
    data = pd.read_csv("fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = np.asarray(pixel_sequence.split(
            ' '), dtype=np.uint8).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]

    return train_faces, train_emotions, val_faces, val_emotions


# loading dataset
train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape

train_faces /= 255.
val_faces /= 255.

# Define the model here, CHANGEME
model = Sequential()
model.add(Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# log the number of total parameters
config.total_params = model.count_params()
model.fit(train_faces, train_emotions, batch_size=config.batch_size,
          epochs=config.num_epochs, verbose=1, callbacks=[
              Perf(val_faces),
              WandbCallback(data_type="image", labels=[
                            "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
          ], validation_data=(val_faces, val_emotions))

# save the model
model.save("emotion.h5")
