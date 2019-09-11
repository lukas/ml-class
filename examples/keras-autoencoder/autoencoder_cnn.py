from util import Images
import wandb
import tensorflow as tf
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.epochs = 30

(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(
    12, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Reshape((28, 28)))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, X_train,
          epochs=config.epochs,
          validation_data=(X_test, X_test),
          callbacks=[Images(X_test), wandb.keras.WandbCallback(save_model=False)])


model.save('auto-cnn.h5')
