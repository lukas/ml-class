# Import layers
import tensorflow as tf
import pandas as pd
import numpy as np
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


class Perf(tf.keras.callbacks.Callback):
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
    if not os.path.exists('face_cache.npz'):
        data = pd.read_csv("fer2013/fer2013.csv")
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = np.asarray(pixel_sequence.split(
                ' '), dtype=np.uint8).reshape(width, height)
            faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()

        val_faces = faces[int(len(faces) * 0.8):]
        val_emotions = emotions[int(len(faces) * 0.8):]
        train_faces = faces[:int(len(faces) * 0.8)]
        train_emotions = emotions[:int(len(faces) * 0.8)]
        np.savez('face_cache.npz', train_faces=train_faces, train_emotions=train_emotions,
                 val_faces=val_faces, val_emotions=val_emotions)
    cached = np.load('face_cache.npz')

    return cached['train_faces'], cached['train_emotions'], cached['val_faces'], cached['val_emotions']


# loading dataset
train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape

train_faces /= 255.
val_faces /= 255.

# Define the model here, CHANGEME
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=input_shape))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
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
