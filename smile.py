# much of the code borrowed and modified from https://github.com/kylemcdonald/SmileCNN
# need to do wget https://github.com/hromi/SMILEsmileD/archive/master.zip

import numpy as np
from skimage.measure import block_reduce
from skimage.io import imread
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from glob import glob
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
X = X[indices]
y = y[indices]

# prepare weighting for classes since they're unbalanced
class_totals = y.sum(axis=0)
class_weight = class_totals.max() / class_totals

img_rows, img_cols = X.shape[1:]

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(img_rows,img_cols,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=128, class_weight=class_weight, nb_epoch=5, verbose=1, validation_split=0.1)
