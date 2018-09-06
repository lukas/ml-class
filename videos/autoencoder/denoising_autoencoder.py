from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model, Sequential
from keras.callbacks import Callback
from keras.datasets import mnist
import numpy as np
import wandb
from wandb.keras import WandbCallback

def add_noise(x_train, x_test):
    # Add some random noise to an image

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy, x_test_noisy

run = wandb.init()
config = run.config

config.encoding_dim = 32
config.epochs = 10

(x_train, _), (x_test, _) = mnist.load_data()
(x_train_noisy, x_test_noisy) = add_noise(x_train, x_test)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(config.encoding_dim, activation='relu'))
model.add(Dense(784, activation='sigmoid'))
model.add(Reshape((28,28)))
model.compile(optimizer='adam', loss='mse')

#for visualization
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


model.fit(x_train_noisy, x_train,
                epochs=config.epochs,
                validation_data=(x_test_noisy, x_test), callbacks=[Images(), WandbCallback()])


model.save("auto-denoise.h5")




