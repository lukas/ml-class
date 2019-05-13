import tensorflow as tf  # pylint: disable=no-name-in-module
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, TimeDistributed, Reshape, Input, GRU, CuDNNGRU, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from datasets import LinesDataset, Generator
from util import ctc_decode, format_batch_ctc, slide_window, ExampleLogger
import wandb

wandb.init()
wandb.config.model = "ctc"
wandb.config.window_width = 28
wandb.config.window_stride = 14

# Load our dataset
dataset = LinesDataset(subsample_fraction=1)
dataset.load_or_generate_data()
image_height, image_width = dataset.input_shape
output_length, num_classes = dataset.output_shape

# Use the correct GRU
gru_fn = CuDNNGRU if tf.test.is_gpu_available() else GRU

# Setup inputs
image_input = Input(shape=dataset.input_shape, name='image_input')
y_true = Input(shape=(output_length,), name='y_true')
input_length = Input(shape=(1,), name='input_length')
label_length = Input(shape=(1,), name='label_length')

# Configure windows over the input image
num_windows = int(
    (image_width - wandb.config.window_width) / wandb.config.window_stride) + 1
if num_windows < output_length:
    raise ValueError(
        f'Window width/stride need to generate >= {output_length} windows (currently {num_windows})')

# Build windowing layer
image_reshaped = Reshape((image_height, image_width, 1))(image_input)
image_patches = Lambda(
    slide_window,
    arguments={'window_width': wandb.config.window_width,
               'window_stride': wandb.config.window_stride}
)(image_reshaped)

# Build simple convnet
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Dropout(0.4))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Dropout(0.4))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))

# TimeDistribute convnet output
cnn_out = TimeDistributed(cnn)(image_patches)

# Feed Time distributed into a GRU
gru_out = Bidirectional(gru_fn(128, return_sequences=True))(cnn_out)
gru2_out = gru_fn(128, return_sequences=True)(gru_out)

# Convert GRU output to our classes
softmax_out = Dense(num_classes, activation='softmax',
                    name='softmax_output')(gru2_out)

# Total number of pixels entering the network
input_length_processed = Lambda(lambda x, num_windows=None: x * num_windows,
                                arguments={'num_windows': num_windows})(input_length)

# Use tensorflow to calculate CTC loss
ctc_loss_output = Lambda(lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]), name='ctc_loss')(
    [y_true, softmax_out, input_length_processed, label_length])

# Out decoded CTC output
ctc_decoded_output = Lambda(lambda x: ctc_decode(x[0], x[1], output_length), name='ctc_decoded')(
    [softmax_out, input_length_processed])

# Build the complete model
complete_model = Model(inputs=[image_input, y_true, input_length, label_length],
                       outputs=[ctc_loss_output, ctc_decoded_output])

# Define loss as our custom ctc_loss output layer
complete_model.compile(
    loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer="adam")

complete_model.summary()


def decode_examples(images, labels):
    """Transform our output for example logging"""
    return complete_model.predict(format_batch_ctc(images, labels)[0])[1]


# Initialize generators and train
train = Generator(dataset.x_train, dataset.y_train,
                  batch_size=32, format_fn=format_batch_ctc)
test = Generator(dataset.x_test, dataset.y_test,
                 batch_size=32, format_fn=format_batch_ctc)
complete_model.fit_generator(
    train,
    epochs=75,
    callbacks=[ExampleLogger(dataset, decode_examples), ModelCheckpoint(
        "best-ctc.h5", save_best_only=True)],
    validation_data=test,
)
