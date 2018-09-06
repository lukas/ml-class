from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.callbacks import Callback

import numpy as np
from util import Images
import wandb
from wandb.keras import WandbCallback
class Images(Callback):
    def on_epoch_end(self, epoch, logs):
        indices = np.random.randint(self.validation_data[0].shape[0], size=8)
        test_data = self.validation_data[0][indices]
        pred_data = self.model.predict(test_data)
        wandb.log({
             "examples": [
                   wandb.Image(np.hstack([data, pred_data[i]]), caption=str(i))
                   for i, data in enumerate(test_data)]
        }, commit=False)

run = wandb.init()
config = run.config

config.epochs = 10

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

model = Sequential()
model.add(Reshape((28,28,1), input_shape=(28,28)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
model.add(Reshape((28,28)))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, X_train,
          epochs=config.epochs,
          validation_data=(X_test, X_test), 
          callbacks=[Images(), WandbCallback(save_model=False)])


model.save('auto-cnn.h5')