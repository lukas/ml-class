from keras.layers import Input, Dense, Flatten, Reshape, UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Model, Sequential

from keras.datasets import mnist
from keras.datasets import fashion_mnist

from keras.callbacks import Callback
import numpy as np
import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.encoding_dim = 32
config.epochs = 1000

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model = Sequential()
model.add(Reshape((28,28,1),input_shape=(28,28)))
model.add(Conv2D(4, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(8, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation="relu"))
model.add(Dense(7*7, activation="relu"))
model.add(Reshape((7,7,1)))
model.add(Conv2D(8, (3,3), padding="same", activation="relu"))
model.add(UpSampling2D())
model.add(Conv2D(4, (3,3), padding="same", activation="relu"))
model.add(UpSampling2D())
model.add(Conv2D(1, (3,3), padding="same", activation="sigmoid"))
model.add(Reshape((28,28)))

model.compile(optimizer='adam', loss='mse')

model.summary()

class Images(Callback):
      def on_epoch_end(self, epoch, logs):
            indices = np.random.randint(self.validation_data[0].shape[0], size=8)
            test_data = self.validation_data[0][indices]
            pred_data = self.model.predict(test_data)
            run.history.row.update({
                  "examples": [
                        wandb.Image(np.hstack([data, pred_data[i]]), caption=str(i))
                        for i, data in enumerate(test_data)]
            })

model.fit(x_train, x_train,
                epochs=config.epochs,
                validation_data=(x_test, x_test), 
                callbacks=[Images(), WandbCallback()])


model.save('auto-small.h5')


