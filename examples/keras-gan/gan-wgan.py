import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import os

# WGAN parameters
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
clip_value = 0.01  # weight clipping range
n_critic = 5       # train the critic more than generator

(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

# Labels not used in WGAN
real = -np.ones((64, 1))
fake = np.ones((64, 1))


def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    return model


def build_critic():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1))
    return model


def sample_images(generator, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    if not os.path.exists("images"):
        os.makedirs("images")
    fig.savefig(f"images/wgan_epoch_{epoch}.png")
    plt.close()


# Build and compile models
optimizer = RMSprop(learning_rate=0.00005)

critic = build_critic()
critic.compile(loss=wgan_loss, optimizer=optimizer)

generator = build_generator()

z = Input(shape=(latent_dim,))
img = generator(z)
critic.trainable = False
validity = critic(img)
combined = Model(z, validity)
combined.compile(loss=wgan_loss, optimizer=optimizer)


# Custom WGAN loss
def wgan_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)


# Training Loop
def train(epochs, batch_size=64, sample_interval=200):
    for epoch in range(epochs):

        for _ in range(n_critic):
            # Train Critic
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_imgs = generator.predict(noise)

            d_loss_real = critic.train_on_batch(imgs, real)
            d_loss_fake = critic.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Clip weights
            for layer in critic.layers:
                weights = layer.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                layer.set_weights(weights)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, real)

        # Log progress
        print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")

        if epoch % sample_interval == 0:
            sample_images(generator, epoch)


if __name__ == "__main__":
    train(epochs=5000, batch_size=64, sample_interval=500)
