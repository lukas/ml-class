import tensorflow as tf  # pylint: disable=no-name-in-module
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, Reshape, Input, CuDNNGRU, TimeDistributed
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from datasets import LinesDataset, Generator
from util import ctc_decode, format_batch_ctc, slide_window, ExampleLogger
import wandb

wandb.init()
wandb.config.model = "cnn"
wandb.config.window_width = 14
wandb.config.window_stride = 7

# Load our dataset
dataset = LinesDataset(subsample_fraction=1)
dataset.load_or_generate_data()
image_height, image_width = dataset.input_shape
output_length, num_classes = dataset.output_shape

model = Sequential()
model.add(Reshape((image_height, image_width, 1),
                  input_shape=dataset.input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

# We are going to use a Conv2D to slide over these outputs with window_width and window_stride,
# and output softmax activations of shape (output_length, num_classes)./
# In your calculation of the necessary filter size,
# remember that padding is set to 'valid' (by default) in the network above.

# (image_height // 2 - 2, image_width // 2 - 2, 64)
new_height = image_height // 2 - 2
new_width = image_width // 2 - 2
new_window_width = wandb.config.window_width // 2 - 2
new_window_stride = wandb.config.window_stride // 2
model.add(Conv2D(128, (new_height, new_window_width),
                 (1, new_window_stride), activation='relu'))
model.add(Dropout(0.3))
# (1, num_windows, 128)

num_windows = int((new_width - new_window_width) / new_window_stride) + 1

model.add(Reshape((num_windows, 128, 1)))

width = int(num_windows / output_length)
model.add(Conv2D(num_classes, (width, 128), (width, 1), activation='softmax'))

model.add(Lambda(lambda x: tf.squeeze(x, 2)))

# Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
model.add(Lambda(lambda x: x[:, :output_length, :]))

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.summary()

# Initialize generators and train
train = Generator(dataset.x_train, dataset.y_train,
                  batch_size=32, augment_fn=None)
test = Generator(dataset.x_test, dataset.y_test,
                 batch_size=32, augment_fn=None)
model.fit_generator(
    train,
    epochs=30,
    callbacks=[ExampleLogger(dataset), ModelCheckpoint(
        "best-cnn.h5", save_best_only=True)],
    validation_data=test,
)
