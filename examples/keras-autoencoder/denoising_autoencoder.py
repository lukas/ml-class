import numpy as np
import tensorflow as tf
import wandb
from util import Images


# Add random noise to images.
def add_noise(x_train, x_test):
    noise_factor = 0.5
    x_train_noisy = x_train + \
        np.random.normal(loc=0.0, scale=noise_factor, size=x_train.shape)
    x_test_noisy = x_test + \
        np.random.normal(loc=0.0, scale=noise_factor, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy, x_test_noisy


# Set Hyper-parameters
run = wandb.init()
config = run.config
config.encoding_dim = 32
config.epochs = 10

# Load and normalize data
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
(X_train_noisy, X_test_noisy) = add_noise(X_train, X_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(config.encoding_dim, activation='relu'))
model.add(tf.keras.layers.Dense(784, activation='sigmoid'))
model.add(tf.keras.layers.Reshape((28, 28)))
model.compile(optimizer='adam', loss='binary_crossentropy')


model.fit(X_train_noisy, X_train,
          epochs=config.epochs,
          validation_data=(X_test_noisy, X_test),
          callbacks=[Images(X_test_noisy), wandb.keras.WandbCallback(save_model=False)])


model.save("auto-denoise.h5")
