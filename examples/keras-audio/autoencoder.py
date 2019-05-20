from keras import backend as K
from keras.callbacks import Callback, TensorBoard
from keras.models import Model, Sequential, load_model
from keras import layers
import numpy as np
import glob
import audio_utilities
import wandb
from wandb.keras import WandbCallback
from keras import optimizers
import os
wandb.init()

sample_rate = 8000
audio_utilities.ensure_audio()
if not os.path.exists("cache"):
    raise ValueError("You must run python preprocess before training")


def load_data():
    print("Loading data...")
    data = {}
    X_train, y_train = [], []
    clean = np.load('cache/clean.npy')
    data['clean'] = clean[:, 1]
    data['dirty'] = []
    for f in glob.glob("cache/*"):
        spects = np.load(f)
        key = f.split("/")[-1]
        data['dirty'].extend(spects[:, 1])
        if key == "clean":
            continue
        else:
            X_train.extend(spects[:, 0])
            y_train.extend(clean[:, 0])
    return np.expand_dims(np.array(X_train), -1), np.expand_dims(np.array(y_train), -1), data


class Audio(Callback):
    def on_epoch_end(self, *args):
        validation_X = self.validation_data[0]
        validation_y = self.validation_data[1]
        val_scales = scales["dirty"][:20]
        validation_length = len(validation_X)
        indices = np.random.choice(
            validation_length, 1, replace=False)
        predictions = self.model.predict(validation_X[indices])
        print("Min: ", predictions.min(), "Max: ", predictions.max())
        predictions = predictions.clip(0, 1)  # np.max(abs(predictions))
        norm_pred = []
        norm_in = []
        clean_in = []
        for i, idx in enumerate(indices):
            scale = val_scales[idx]
            pred = np.squeeze(predictions[i])
            norm = np.squeeze(validation_X[idx])
            clean = np.squeeze(validation_y[idx])
            norm_pred.append(audio_utilities.griffin_lim(pred, scale))
            norm_in.append(audio_utilities.griffin_lim(norm, scale))
            clean_in.append(audio_utilities.griffin_lim(clean, scale))

        wandb.log({
            "clean_audio": [wandb.Audio(audio, sample_rate=sample_rate) for audio in clean_in],
            "noisy_audio": [wandb.Audio(audio, sample_rate=sample_rate) for audio in norm_in],
            "audio": [wandb.Audio(audio, sample_rate=sample_rate)
                      for audio in norm_pred]}, commit=False)


X_train, y_train, scales = load_data()

model = Sequential()
model.add(layers.Conv2D(64, 3, padding="same",
                        activation="relu", use_bias=False, input_shape=(X_train[0].shape)))
model.add(layers.Conv2D(32, 3, strides=(2, 2),
                        padding="same", activation="relu", use_bias=False))
model.add(layers.Conv2D(2, 3, strides=(2, 2),
                        padding="same", activation="relu", use_bias=False))
model.add(layers.UpSampling2D(2))
model.add(layers.Conv2D(32, 3, padding="same",
                        activation="relu", use_bias=False))
model.add(layers.UpSampling2D(2))
model.add(layers.Conv2D(64, 3, padding="same",
                        activation="relu", use_bias=False))
model.add(layers.Conv2D(1, 3, padding="same", use_bias=False))
model.add(layers.Cropping2D(cropping=((0, 0), (0, 3))))
model.summary()

model.compile(optimizer="adagrad", loss="mse")

model.fit(X_train[20:], y_train[20:], epochs=60, batch_size=64,
          validation_data=(X_train[:20], y_train[:20]), callbacks=[Audio(), WandbCallback(data_type="image")])
