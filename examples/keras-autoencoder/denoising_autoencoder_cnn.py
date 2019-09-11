import numpy as np
import wandb
import tensorflow as tf
from util import Images


# Add noise
def add_noise(x_train, x_test):
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * \
        np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * \
        np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy, x_test_noisy


# Set Hyper-parameters
run = wandb.init()
config = run.config
config.encoding_dim = 32
config.epochs = 10

# Load and normalize data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
(x_train_noisy, x_test_noisy) = add_noise(x_train, x_test)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(tf.keras.layers.Conv2D(
    32, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(
    16, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.UpSampling2D())
model.add(tf.keras.layers.Conv2D(
    1, (3, 3), padding='same', activation='sigmoid'))
model.add(tf.keras.layers.Reshape((28, 28)))
model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noisy, x_train,
          epochs=config.epochs,
          validation_data=(x_test_noisy, x_test),
          callbacks=[Images(x_test_noisy), wandb.keras.WandbCallback(save_model=False)])

model.save("auto-denoise.h5")
