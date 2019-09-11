(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

model = tf.keras.models.load_model("perceptron.h5")

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print(model.layers[1].get_weights())
