# This is a denoising autoencoder
#
# We feed in fashion mnist data (b&w photos of apparel) with noise added.
# The goal is to remove the noise and return the original image.
# In this case we are using mse (mean squared error) as the loss function.
#
# More on denoising autoencoders here: https://towardsdatascience.com/denoising-autoencoders-explained-dbb82467fc2
#
# Can you get the mean squared error of the reconstructed image under 0.03 on 
# the validation data?  There are multiple ways to do it, but one cool way
# is to add convolutional layers.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, Dropout, UpSampling2D
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import wandb
from wandb.keras import WandbCallback


# logging code
run = wandb.init(project="denoising-autoencoder")
config = run.config
config.encoding_dim = 32

def add_noise(x_train, x_test):
    # Function to add some random noise
    noise_factor = 1.0
    x_train_noisy = x_train + np.random.normal(loc=0.0, scale=noise_factor, size=x_train.shape) 
    x_test_noisy = x_test + np.random.normal(loc=0.0, scale=noise_factor, size=x_test.shape) 
    
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy, x_test_noisy

def log_images(epoch, logs):
    # Function to show the before and after images at each step
    indices = np.random.randint(x_test_noisy.shape[0], size=8)
    test_data = x_test_noisy[indices]
    pred_data = np.clip(model.predict(test_data), 0, 1)
    wandb.log({
            "examples": [
                wandb.Image(np.hstack([data, pred_data[i]]), caption=str(i))
                for i, data in enumerate(test_data)]
        }, commit=False)

    
(x_train, _), (x_test, _) = fashion_mnist.load_data()
(x_train_noisy, x_test_noisy) = add_noise(x_train, x_test)
img_width = x_train.shape[1]
img_height = x_train.shape[2]

x_train = x_train / 255.
x_test = x_test / 255.

# create model
model = Sequential()
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(config.encoding_dim, activation="relu"))
model.add(Dense(img_width*img_height, activation="sigmoid"))
model.add(Reshape((img_width, img_height)))
model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])

# Fit the model
model.fit(x_train_noisy, x_train, epochs=30, validation_data=(x_test_noisy, x_test),
          callbacks=[WandbCallback(), LambdaCallback(on_epoch_end=log_images)])
