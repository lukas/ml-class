import tensorflow as tf
import wandb

wandb.init()

config = wandb.config
config.batch_size = 128
config.epochs = 10
config.learn_rate = 1000

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Dense(num_classes))
model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(config.learn_rate),
              metrics=['accuracy'])
# log the number of total parameters
config.total_params = model.count_params()
print("Total params: ", config.total_params)
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test),
          callbacks=[wandb.keras.WandbCallback(data_type="image", labels=class_names, save_model=False)])
