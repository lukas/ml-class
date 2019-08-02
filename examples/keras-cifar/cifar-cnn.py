from keras.callbacks import TensorBoard
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np
import os
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from keras import backend as K


run = wandb.init()
config = run.config
config.dropout = 0.25
config.dense_layer_nodes = 100
config.learn_rate = 0.08
config.batch_size = 128
config.epochs = 10

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(config.dropout))

model.add(Flatten())
model.add(Dense(config.dense_layer_nodes, activation='relu'))
model.add(Dropout(config.dropout))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

datagen = ImageDataGenerator(width_shift_range=0.1)
datagen.fit(X_train)


class CollectOutputAndTarget(keras.callbacks.Callback):
    def __init__(self, model, class_names):
        super(CollectOutputAndTarget, self).__init__()
        self.class_names = class_names

        self.highest_train_losses = []
        self.highest_train_loss_inputs = []
        self.highest_train_loss_outputs = []
        self.highest_train_loss_targets = []

        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)
        self.var_x = tf.Variable(0., validate_shape=False)
        self.var_loss = tf.Variable(0., validate_shape=False)

        # TODO: Handle weighted loss functions, multiple loss functions
        individual_loss = model.loss_functions[0](
            model.targets[0], model.outputs[0])

        # TODO: handle multiple outputs
        fetches = [tf.assign(self.var_x, model.inputs[0], validate_shape=False),
                   tf.assign(self.var_y_true,
                             model.targets[0], validate_shape=False),
                   tf.assign(self.var_y_pred,
                             model.outputs[0], validate_shape=False),
                   tf.assign(self.var_loss, individual_loss,
                             validate_shape=False),
                   ]

        model._function_kwargs = {'fetches': fetches}

    def on_batch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        targets = K.eval(self.var_y_true)
        outputs = K.eval(self.var_y_pred)
        inputs = K.eval(self.var_x)
        losses = K.eval(self.var_loss)

        if (len(self.highest_train_losses) > 0):
            self.highest_train_losses = np.append(
                self.highest_train_losses, losses, axis=0)
            self.highest_train_loss_inputs = np.append(
                self.highest_train_loss_inputs, inputs, axis=0)
            self.highest_train_loss_outputs = np.append(
                self.highest_train_loss_outputs, outputs, axis=0)
            self.highest_train_loss_targets = np.append(
                self.highest_train_loss_targets, targets, axis=0)
        else:
            self.highest_train_losses = losses
            self.highest_train_loss_inputs = inputs
            self.highest_train_loss_outputs = outputs
            self.highest_train_loss_targets = targets

        top_idx = np.argsort(self.highest_train_losses)[-10:]
        self.highest_train_losses = self.highest_train_losses[top_idx]
        self.highest_train_loss_inputs = self.highest_train_loss_inputs[top_idx]
        self.highest_train_loss_outputs = self.highest_train_loss_outputs[top_idx]
        self.highest_train_loss_targets = self.highest_train_loss_targets[top_idx]

    def on_epoch_end(self, epoch, logs={}):
        top_idx = np.argsort(self.highest_train_losses)[-10:]

        images = []
        for idx in top_idx:
            max_target_idx = np.argmax(self.highest_train_loss_targets[idx])
            max_output_idx = np.argmax(self.highest_train_loss_outputs[idx])
            output_prob = self.highest_train_loss_outputs[idx][max_output_idx]
            images.append(wandb.Image(self.highest_train_loss_inputs[idx],
                                      caption=f"Real: {self.class_names[max_target_idx]} " +
                                      f"Pred: {self.class_names[max_output_idx]} " +
                                      "({0:.2f})".format(output_prob)))
        wandb.log({"highest_loss_examples": images}, commit=False)
        wandb.log(logs)


# lame that we need pass in the model, but we will do some surgery on this model
cbk = CollectOutputAndTarget(model, class_names)

# indicate folder to save, plus other options
tensorboard = TensorBoard(log_dir=wandb.run.dir, histogram_freq=1,
                          write_graph=True, write_images=True)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=config.batch_size),
                    steps_per_epoch=X_train.shape[0] // config.batch_size,
                    epochs=config.epochs,
                    validation_data=(X_test, y_test),
                    workers=2,
                    callbacks=[cbk])
