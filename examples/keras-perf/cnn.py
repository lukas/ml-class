import os
import tensorflow as tf
from tensorflow.python.client import timeline
import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dropout = 0.2
config.dense_layer_size = 128
config.img_width = 32
config.img_height = 32
config.channels = 3
config.epochs = 4

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# reshape input data
# X_train = X_train.reshape(
#    X_train.shape[0], config.img_width, config.img_height, 1)
# X_test = X_test.reshape(
#    X_test.shape[0], config.img_width, config.img_height, 1)

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]
labels = [str(i) for i in range(10)]

# build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,
                                 (config.first_layer_conv_width,
                                  config.first_layer_conv_height),
                                 input_shape=(config.img_width,
                                              config.img_height, config.channels),
                                 activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(config.dense_layer_size, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# optional profiling setup...
# > sudo pip install tensorflow_gpu==1.14.0
#run_options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata = tf.compat.v1.RunMetadata()
# options=run_options, run_metadata=run_metadata,
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# log the number of total parameters
config.total_params = model.count_params()
print("Total params: ", config.total_params)

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=config.epochs,
          callbacks=[WandbCallback(data_type="image", save_model=False),
                     tf.keras.callbacks.TensorBoard(log_dir=wandb.run.dir)])
model.save('cnn.h5')

# optional profiling setup continued
#tl = timeline.Timeline(run_metadata.step_stats)
# with open('profile.trace', 'w') as f:
#    f.write(tl.generate_chrome_trace_format())

# Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file('cnn.h5')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open("cnn.tflite", "wb").write(tflite_model)
