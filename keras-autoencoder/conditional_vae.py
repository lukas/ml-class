from wandb.keras import WandbCallback
import wandb
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.callbacks import Callback
from keras.models import Model, load_model
from keras import layers
import keras
import plotly.graph_objs as go
import plotly.plotly as py
from sklearn import manifold
import matplotlib.pyplot as plt
import pdb
import os
import subprocess
import pandas as pd
import numpy as np
import cv2
import sys
import matplotlib
matplotlib.use("Agg")


wandb.init(project="cvae")
wandb.config.latent_dim = 16
wandb.config.labels = ["1", "7"]  # ["Happy", "Sad"]
wandb.config.batch_size = 128
wandb.config.epochs = 50
wandb.config.variational = False
wandb.config.conditional = True
wandb.config.dataset = "mnist"

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def load_fer2013(filter_emotions=[]):
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output(
            "curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz", shell=True)
    print("Loading dataset into memory...")
    data = pd.read_csv("fer2013/fer2013.csv")
    if len(filter_emotions) > 0:
        data = data.loc[data["emotion"].isin(
            [EMOTIONS.index(f) for f in filter_emotions])]
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = np.asarray(pixel_sequence.split(
            ' '), dtype=np.uint8).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces) / 255.
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]

    return train_faces, train_emotions, val_faces, val_emotions


def vae_loss(x, t_decoded):
    ''' Negative variational lower bound used as loss function for training the variational auto-encoder. '''
    # Reconstruction loss
    rc_loss = K.binary_crossentropy(x, t_decoded)
    rc_loss *= img_size

    if wandb.config.variational:
        # Regularization term (KL divergence)
        kl_loss = -0.5 * K.sum(1 + t_log_var
                               - K.square(t_mean)
                               - K.exp(t_log_var), axis=-1)
        # Average over mini-batch
        return K.mean(rc_loss + kl_loss)
    else:
        return K.mean(rc_loss)


get_custom_objects().update({"vae_loss": vae_loss})


def sample(args):
    '''
    Draws samples from a standard normal and scales the samples with
    standard deviation of the variational distribution and shifts them
    by the mean.

    Args:
        args: sufficient statistics of the variational distribution.

    Returns:
        Samples from the variational distribution.
    '''
    t_mean, t_log_var = args
    t_sigma = K.sqrt(K.exp(t_log_var))
    epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * epsilon


def concat_label(args):
    '''
    Converts 2d labels to 3d, i.e. [[0,1]] = [[[0,0],[0,0]],[[1,1],[1,1]]]
    '''
    x, labels = args
    x_shape = K.int_shape(x)
    label_shape = K.int_shape(labels)
    output = K.reshape(labels, (K.shape(x)[0],
                                1, 1, label_shape[1]))
    output = K.repeat_elements(output, x_shape[1], axis=1)
    output = K.repeat_elements(output, x_shape[2], axis=2)

    return K.concatenate([x, output], axis=-1)


class ShowImages(Callback):
    def on_epoch_end(self, epoch, logs):
        indicies = np.random.randint(X_test.shape[0], size=36)
        tsne_idx = np.random.randint(X_test.shape[0], size=500)
        inputs = X_test[indicies]
        t_inputs = X_train[indicies]
        r_labels = y_test[indicies]
        labels = keras.utils.to_categorical(
            np.random.randint(len(wandb.config.labels), size=36))
        t_labels = y_train[indicies]

        results = vae.predict([inputs, r_labels, labels])
        t_results = vae.predict([t_inputs, t_labels, t_labels])
        if wandb.config.variational:
            output = encoder.predict([X_test[tsne_idx], y_test[tsne_idx]])
            latent = K.eval(sample(output))  # output[0]
        else:
            latent = encoder.predict([X_test[tsne_idx], y_test[tsne_idx]])
        # Compute t-SNE embedding of latent space
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X = tsne.fit_transform(latent)
        X_tsne = X
        trace = go.Scatter(x=list(X_tsne[:, 0]), y=list(X_tsne[:, 1]),
                           mode='markers', showlegend=False,
                           marker=dict(color=list(np.argmax(y_test[tsne_idx], axis=1)),
                                       colorscale='Viridis',
                                       size=8,
                                       showscale=True))
        fig = go.Figure(data=[trace])
        wandb.log({"latent_tsne": fig}, commit=False)
        wandb.log({
            "train_images": [wandb.Image(
                np.hstack([t_inputs[i], res])) for i, res in enumerate(t_results)
            ]
        }, commit=False)
        wandb.log({
            "images": [wandb.Image(
                np.hstack([inputs[i], res]), caption=" to ".join([
                    wandb.config.labels[np.argmax(r_labels[i])
                                        ], wandb.config.labels[np.argmax((labels)[i])]
                ])) for i, res in enumerate(results)]}, commit=False)


def create_encoder(input_shape):
    encoder_input = layers.Input(shape=input_shape)
    label_input = layers.Input(shape=(len(wandb.config.labels),))
    if wandb.config.conditional:
        x = layers.Lambda(concat_label, name="c")([encoder_input, label_input])
    else:
        x = encoder_input
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same',
                      activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same',
                      activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(wandb.config.latent_dim * 2, activation='relu')(x)
    if wandb.config.variational:
        t_mean = layers.Dense(wandb.config.latent_dim, name="latent_mean")(x)
        t_log_var = layers.Dense(
            wandb.config.latent_dim, name="latent_variance")(x)
        output = [t_mean, t_log_var]
    else:
        output = layers.Dense(wandb.config.latent_dim, activation="relu")(x)

    return Model([encoder_input, label_input], output, name='encoder')


def create_categorical_decoder():
    decoder_input = layers.Input(shape=(wandb.config.latent_dim,))
    label_input = layers.Input(shape=(len(wandb.config.labels),))
    if wandb.config.conditional:
        x = layers.concatenate([decoder_input, label_input], axis=1)
    else:
        x = decoder_input
    x = layers.Dense(img_size // 2 * img_size // 2 * 64, activation='relu')(x)
    x = layers.Reshape((img_size // 2, img_size // 2, 64))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(
        32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    return Model([decoder_input, label_input], x, name='decoder')


if wandb.config.dataset == "emotions":
    img_size = 48
    X_train, y_train, X_test, y_test = load_fer2013(
        filter_emotions=wandb.config.labels)
else:
    img_size = 28
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_train /= 255.
    X_test = X_test.astype('float32')
    X_test /= 255.

    # reshape input data
    X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
    X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 1)

    labels = [int(l) for l in wandb.config.labels]
    X_train = X_train[np.isin(y_train, labels)]
    X_test = X_test[np.isin(y_test, labels)]
    filtered_train = y_train[np.isin(y_train, labels)]
    filtered_test = y_test[np.isin(y_test, labels)]
    for i, label in enumerate(labels):
        filtered_train = np.where(filtered_train == label, i, filtered_train)
        filtered_test = np.where(filtered_test == label, i, filtered_test)
    y_train = keras.utils.to_categorical(filtered_train)
    y_test = keras.utils.to_categorical(filtered_test)

encoder = create_encoder(input_shape=(img_size, img_size, 1))
decoder = create_categorical_decoder()
sampler = layers.Lambda(sample, name='sampler')

image = layers.Input(shape=(img_size, img_size, 1))
true_label = layers.Input(shape=(len(wandb.config.labels),))
dest_label = layers.Input(shape=(len(wandb.config.labels),))
output = encoder([image, true_label])
if wandb.config.variational:
    t_mean, t_log_var = output
    t = sampler(output)
else:
    t = output
t_decoded = decoder([t, dest_label])

vae = Model([image, true_label, dest_label], t_decoded, name='vae')
vae.compile(optimizer='rmsprop', loss=vae_loss)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
vae.fit([X_train, y_train, y_train], X_train, epochs=wandb.config.epochs,
        shuffle=True, batch_size=wandb.config.batch_size, callbacks=[ShowImages(), WandbCallback()],
        validation_data=([X_test, y_test, y_test], X_test))
encoder.save("encoder.h5")
decoder.save("decoder.h5")
