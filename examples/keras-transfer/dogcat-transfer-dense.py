import os
import sys
import glob
import argparse
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from dogcat_data import generators, get_nb_files

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config
#fixed size for DenseNet121
config.img_width = 224
config.img_height = 224
config.epochs = 10
config.batch_size = 128

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    optimizer = Adam(lr=0.0001,
                     beta_1=0.9,
                     beta_2=0.999,
                     epsilon=None,
                     decay=0.0)
    sgd = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes, activation='softmax'):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    predictions = Dense(nb_classes, activation=activation)(base_model.output)
    return Model(inputs=base_model.input, outputs=predictions)

train_dir = "dogcat-data/train"
val_dir = "dogcat-data/validation"
nb_train_samples = get_nb_files(train_dir)
nb_classes = len(glob.glob(train_dir + "/*"))
nb_val_samples = get_nb_files(val_dir)

train_generator, validation_generator = generators(preprocess_input, config.img_width, config.img_height, config.batch_size)

# setup model
base_model = DenseNet121(input_shape=(config.img_width, config.img_height, 3),
                         weights='imagenet',
                         include_top=False,
                         pooling='avg')

model = add_new_last_layer(base_model, nb_classes)

# transfer learning
setup_to_transfer_learn(model, base_model)

history_tl = model.fit_generator(
    train_generator,
    workers=2,
    epochs=config.epochs,
    steps_per_epoch=nb_train_samples * 2 / config.batch_size,
    validation_data=validation_generator,
    validation_steps=nb_train_samples / config.batch_size,
    callbacks=[WandbCallback(data_type="image", generator=validation_generator, labels=['cat', 'dog'], save_model=False)],
    class_weight='auto')

model.save("transfered.h5")
