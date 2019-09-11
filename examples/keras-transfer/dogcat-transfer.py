import os
import sys
import glob
import argparse
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from dogcat_data import generators, get_nb_files

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config
config.img_width = 299
config.img_height = 299
config.epochs = 5
config.fc_size = 1024
config.batch_size = 128


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
      Args:
        base_model: keras model excluding top
        nb_classes: # of classes
      Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = Dense(config.fc_size, activation='relu')(
        x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(
        x)  # new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


train_dir = "dogcat-data/train"
val_dir = "dogcat-data/validation"

nb_train_samples = get_nb_files(train_dir)
nb_classes = len(glob.glob(train_dir + "/*"))
nb_val_samples = get_nb_files(val_dir)

# data prep
train_generator, validation_generator = generators(
    preprocess_input, config.img_width, config.img_height, config.batch_size)

# setup model
base_model = InceptionV3(weights='imagenet', include_top=False, pooling="avg")
model = add_new_last_layer(base_model, nb_classes)
model._is_graph_network = False

# transfer learning
setup_to_transfer_learn(model, base_model)

model.fit_generator(
    train_generator,
    epochs=config.epochs,
    workers=2,
    steps_per_epoch=nb_train_samples * 2 / config.batch_size,
    validation_data=validation_generator,
    validation_steps=nb_train_samples / config.batch_size,
    callbacks=[WandbCallback(data_type="image", generator=validation_generator, labels=[
                             'cat', 'dog'], save_model=False)],
    class_weight='auto')

model.save('transfered.h5')
