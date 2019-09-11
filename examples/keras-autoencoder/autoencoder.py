from util import Images
import tensorflow as tf
import wandb

run = wandb.init()
config = run.config

config.encoding_dim = 10
config.epochs = 10

(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

encoder = tf.keras.models.Sequential()
encoder.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
encoder.add(tf.keras.layers.Dense(128, activation="relu"))
encoder.add(tf.keras.layers.Dense(64, activation="relu"))
encoder.add(tf.keras.layers.Dense(config.encoding_dim, activation="relu"))

decoder = tf.keras.models.Sequential()
decoder.add(tf.keras.layers.Dense(64, activation="relu",
                                  input_shape=(config.encoding_dim,)))
decoder.add(tf.keras.layers.Dense(128, activation="relu"))
decoder.add(tf.keras.layers.Dense(28*28, activation="sigmoid"))
decoder.add(tf.keras.layers.Reshape((28, 28)))

model = tf.keras.models.Sequential()
model.add(encoder)
model.add(decoder)

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, X_train,
          epochs=config.epochs,
          validation_data=(X_test, X_test),
          callbacks=[Images(X_test), wandb.keras.WandbCallback(save_model="false")])

encoder.save('auto-encoder.h5')
decoder.save('auto-decoder.h5')
