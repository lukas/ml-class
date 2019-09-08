import tensorflow as tf
import wandb

# logging code
run = wandb.init()

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

is_five_train = y_train == 5
is_five_test = y_test == 5
labels = ["Not Five", "Is Five"]

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(img_width, img_height)))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, is_five_train, epochs=3, validation_data=(X_test, is_five_test),
          callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels, save_model=False)])
model.save('perceptron.h5')
