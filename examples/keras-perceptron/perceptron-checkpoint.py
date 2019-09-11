import tensorflow as tf
import wandb

wandb.init()

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datsets.mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# normalize data
X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
num_classes = y_train.shape[1]

# create model
model = tf.keras.models.Sequential()
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          callbacks=[checkpoint, wandb.keras.WandbCallback])
