import os
import numpy as np

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from PIL import Image
from keras.callbacks import LambdaCallback
import wandb

# Find more tricks here: https://github.com/soumith/ganhacks

run = wandb.init()
config = wandb.config

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 10

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)

config.lr=0.0002
config.beta_1=0.5
config.batch_size=128
config.epochs=10

# Optimizer
adam = Adam(config.lr, beta_1=config.beta_1)

generator = Sequential()
generator.add(Dense(256, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics = ['binary_accuracy'])

iter = 0
# Write out generated MNIST images
def writeGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)
    
    imgs = [None] * 10
    for i in range(10):
        imgs[i] = Image.fromarray((generatedImages[0] + 1.)* (255/2.))
        imgs[i] = imgs[i].convert('RGB')
        imgs[i] = (wandb.Image(imgs[i]))
    wandb.log({"image": imgs}, commit=False)


# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)


def log_generator(epoch, logs):
    global iter
    iter += 1
    if iter % 500 == 0:
        wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['binary_accuracy'],
                     'discriminator_loss': 0.0,
                     'discriminator_acc': (1-logs['binary_accuracy'])})

def log_discriminator(epoch, logs):
    global iter
    if iter % 500 == 250:
        wandb.log({
            'generator_loss': 0.0,
            'generator_acc': logs['binary_accuracy'],
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['binary_accuracy']})

def train(epochs=config.epochs, batchSize=config.batch_size):
    batchCount = int(X_train.shape[0] / config.batch_size)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    wandb_logging_callback_d = LambdaCallback(on_epoch_end=log_discriminator)
    wandb_logging_callback_g = LambdaCallback(on_epoch_end=log_generator)


    for e in range(1, epochs+1):
        print("Epoch {}:".format(e))
        for i in range(batchCount):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.fit(X, yDis, verbose=0, callbacks=[wandb_logging_callback_d])

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.fit(noise, yGen, verbose=0, callbacks=[wandb_logging_callback_g])

            writeGeneratedImages(i)

        print("Discriminator loss: {}, acc: {}".format(dloss.history["loss"][-1], dloss.history["binary_accuracy"][-1]))
        print("Generator loss: {}, acc: {}".format(gloss.history["loss"][-1], 1-gloss.history["binary_accuracy"][-1]))


if __name__ == '__main__':
    train(200, 128)
