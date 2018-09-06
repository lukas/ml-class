from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model, Sequential

from keras.datasets import mnist
from keras.callbacks import Callback
import numpy as np
import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.encoding_dim = 32
config.epochs = 10

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(config.encoding_dim, activation='relu'))
model.add(Dense(28*28, activation='sigmoid'))
model.add(Reshape((28,28)))
model.compile(optimizer='adam', loss='mse')

# For visualization
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


model.save('auto.h5')


