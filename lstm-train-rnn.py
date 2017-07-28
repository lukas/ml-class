
import numpy as np
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, SimpleRNN
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model

with open('book.pkl', 'rb') as input:
	cached_data = pickle.load(input)
	char_to_int = cached_data['char_to_int']
	int_to_char = cached_data['int_to_char']
	X = cached_data['X']
	y = cached_data['y']
	print("Read cache file %s." % input.name)

# define the LSTM model
model = Sequential()
model.add(SimpleRNN(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=2, batch_size=128, callbacks=callbacks_list)
model.save("book.h5")
