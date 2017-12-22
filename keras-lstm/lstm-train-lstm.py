
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
import keras
import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config
config.nodes = 50
config.dropout = 0.1
config.epochs = 1000

with open('book.pkl', 'rb') as input:
	cached_data = pickle.load(input)
	char_to_int = cached_data['char_to_int']
	int_to_char = cached_data['int_to_char']
	X = cached_data['X']
	y = cached_data['y']
	print("Read cache file %s." % input.name)


start = np.random.randint(0, len(X)-1)
pattern = [int(x) for x in X[start] * len(char_to_int)]
#pattern = pattern[:10]
#print("".join(int_to_char[i] for i in pattern))

model = load_model('weights-improvement-01-3.1691.hdf5')

class SampleText(keras.callbacks.Callback):
        def on_epoch_end(self, batch, logs={}):
                self.generate_characters()
                
        def generate_characters(self):
                start = np.random.randint(0, len(X)-1)
                pattern = [int(x) for x in X[start] * len(char_to_int)]

                print("Starting with: \"", ''.join([int_to_char[value] for value in pattern]), "\"")

                # generate characters
                for i in range(100):
	                x = np.reshape(pattern, (1, len(pattern), 1))
	                x = x / float(len(int_to_char))
	                prediction = model.predict(x, verbose=0)
	                #print(prediction[0])
	                index = np.random.choice(range(len(prediction[0])), p = prediction[0]  / sum(prediction[0] ))
	                result = int_to_char[index]
	                seq_in = [int_to_char[value] for value in pattern]
	                sys.stdout.write(result)
	                sys.stdout.flush()
	                pattern.append(index)
	                pattern = pattern[1:len(pattern)]

        
# define the LSTM model
model = Sequential()
model.add(LSTM(config.nodes, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(config.nodes))
model.add(Dropout(config.dropout))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, WandbKerasCallback(), SampleText()]
print("Ready to go")

model.fit(X, y, epochs=config.epochs, batch_size=128, callbacks=callbacks_list)
model.save("book-lstm.h5")
