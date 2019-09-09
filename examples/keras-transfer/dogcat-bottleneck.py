
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from dogcat_data import generators, get_nb_files
import os
import sys
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config

# dimensions of our images.
config.img_width = 224
config.img_height = 224
config.epochs = 50
config.batch_size = 40

top_model_weights_path = 'bottleneck.h5'
train_dir = 'dogcat-data/train'
validation_dir = 'dogcat-data/validation'
nb_train_samples = 1000
nb_validation_samples = 1000


def save_bottlebeck_features():
    if os.path.exists('bottleneck_features_train.npy') and (len(sys.argv) == 1 or sys.argv[1] != "--force"):
        print("Using saved features, pass --force to save new features")
        return
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(config.img_width, config.img_height),
        batch_size=config.batch_size,
        class_mode="binary")

    val_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(config.img_width, config.img_height),
        batch_size=config.batch_size,
        class_mode="binary")

    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet')

    print("Predicting bottleneck training features")
    training_labels = []
    training_features = []
    for batch in range(5):  # nb_train_samples // config.batch_size):
        data, labels = next(train_generator)
        training_labels.append(labels)
        training_features.append(model.predict(data))
    training_labels = np.concatenate(training_labels)
    training_features = np.concatenate(training_features)
    np.savez(open('bottleneck_features_train.npy', 'wb'),
             features=training_features, labels=training_labels)

    print("Predicting bottleneck validation features")
    validation_labels = []
    validation_features = []
    validation_data = []
    for batch in range(nb_validation_samples // config.batch_size):
        data, labels = next(val_generator)
        validation_features.append(model.predict(data))
        validation_labels.append(labels)
        validation_data.append(data)
    validation_labels = np.concatenate(validation_labels)
    validation_features = np.concatenate(validation_features)
    validation_data = np.concatenate(validation_data)
    np.savez(open('bottleneck_features_validation.npy', 'wb'),
             features=validation_features, labels=validation_labels, data=validation_data)


def train_top_model():
    train = np.load(open('bottleneck_features_train.npy', 'rb'))
    X_train, y_train = (train['features'], train['labels'])
    test = np.load(open('bottleneck_features_validation.npy', 'rb'))
    X_test, y_test, val_data = (test['features'], test['labels'], test['data'])

    model = Sequential()
    model.add(Flatten(input_shape=X_train[0].shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    class Images(Callback):
        def on_epoch_end(self, epoch, logs):
            base_model = VGG16(include_top=False, weights='imagenet')
            indices = np.random.randint(val_data.shape[0], size=36)
            test_data = val_data[indices]
            features = base_model.predict(
                np.array([preprocess_input(data) for data in test_data]))
            pred_data = model.predict(features)
            wandb.log({
                "examples": [
                      wandb.Image(
                          test_data[i], caption="cat" if pred_data[i] < 0.5 else "dog")
                      for i, data in enumerate(test_data)]
            }, commit=False)

    model.fit(X_train, y_train,
              epochs=config.epochs,
              batch_size=config.batch_size,
              validation_data=(X_test, y_test),
              callbacks=[Images(), WandbCallback(save_model=False)])
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()
