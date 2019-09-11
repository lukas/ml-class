import tensorflow as tf
import wandb

# initialize wandb & set hyperparamers
run = wandb.init()
config = run.config
config.optimizer = "adam"
config.epochs = 50
config.dropout = 10
config.hidden_nodes = 100

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
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
labels = [str(i) for i in range(10)]
num_classes = y_train.shape[1]


# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(img_width, img_height)))
model.add(tf.keras.layers.Dense(config.hidden_nodes, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=config.optimizer,
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=config.epochs,
          callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels)])
