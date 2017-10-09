# need to download wget https://www.gutenberg.org/files/11/11-0.txt
# mv 11-0.txt book.txt
# other interesting data sets http://www.cs.cmu.edu/Groups/AI/util/areas/nlp/corpora/names/

import numpy as np
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model

filename = "male.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i,c) for (c,i) in char_to_int.items())

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX, dataY = [], []
for i in range(len(raw_text) - seq_length):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)




# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(len(chars))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

print(len(char_to_int))
print(len(int_to_char))
print(X.shape)  # x is array of 100 vals for every sequence
print(y.shape)  # y is a single val but one-hot encoded for every sequence

# save the preprocessed text
cached_data = {
	'char_to_int': char_to_int,
	'int_to_char': int_to_char,
	'X': X,
	'y': y
}
with open('book.pkl', 'wb') as output:
	pickle.dump(cached_data, output)
print("Wrote cache to %s." % output.name)
