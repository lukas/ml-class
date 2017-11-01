# much of the code borrowed and modified from https://github.com/kylemcdonald/SmileCNN
# need to do wget https://github.com/hromi/SMILEsmileD/archive/master.zip

import numpy as np
from skimage.measure import block_reduce
from skimage.io import imread
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from glob import glob

import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config

negative_paths = glob('SMILEsmileD-master/SMILEs/negatives/negatives7/*.jpg')
positive_paths = glob('SMILEsmileD-master/SMILEs/positives/positives7/*.jpg')
examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]

def examples_to_dataset(examples, block_size=2):
    X = []
    y = []
    for path, label in examples:
        img = imread(path, as_grey=True)
        img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
        X.append(img)
        y.append(label)
    return np.asarray(X), np.asarray(y)

X, y = examples_to_dataset(examples)

X = X.astype(np.float32) / 255.
y = y.astype(np.int32)

# convert classes to vector
nb_classes = 2
y = np_utils.to_categorical(y, nb_classes).astype(np.float32)

# shuffle all the data
indices = np.arange(len(X))
np.random.shuffle(indices)

train_cutoff = int(len(X) * 0.9)
print(train_cutoff)

X_train = X[indices[:train_cutoff]]
y_train = y[indices[:train_cutoff]]

X_val = X[indices[train_cutoff:]]
y_val = y[indices[train_cutoff:]]



# prepare weighting for classes since they're unbalanced
class_totals = y.sum(axis=0)
class_weight = class_totals.max() / class_totals

img_rows, img_cols = X.shape[1:]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(img_rows,img_cols,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()
#from pdb import set_trace;set_trace()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(
    #featurewise_center=True,
    #samplewise_center=True,
    rotation_range=2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
print(X_train.shape)

model.fit_generator(datagen.flow(X_train, y_train, batch_size=128), class_weight=class_weight,
                    nb_epoch=1, verbose=1, steps_per_epoch=100,
                    callbacks=[WandbKerasCallback()],
                    validation_data=(X_val, y_val))
model.save("smile.h5")
