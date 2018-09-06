from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import wandb
from wandb.keras import WandbCallback

import numpy as np

wandb.init()
config = wandb.config

config.epochs = 10
config.batch_size = 16

def load_names():
    with open("male.txt") as f:
        m_names = f.readlines()

    with open("female.txt") as f:
        f_names = f.readlines()

    mf_names = []

    # remove the names that are both male and female
    for f_name in f_names:
        if f_name in m_names:
            mf_names.append(f_name)

    m_names = [m_name.lower() for m_name in m_names if not m_name in mf_names]
    f_names = [f_name.lower() for f_name in f_names if not f_name in mf_names]

    return m_names, f_names

m_names, f_names = load_names()
    
totalEntries = len(m_names) + len(f_names)
maxlen = 20

chars = set(  "".join(m_names) + "".join(f_names)  )
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X = np.zeros((totalEntries , maxlen, len(chars) ), dtype=np.float32)
y = np.zeros((totalEntries , 2 ), dtype=np.float32)

print(m_names)

for i, name in enumerate(m_names):
    for t, char in enumerate(name):
        X[i, t, char_indices[char]] = 1
    y[i, 0 ] = 1

for i, name in enumerate(f_names):
    for t, char in enumerate(name):
        X[i + len(m_names), t, char_indices[char]] = 1
    y[i + len(m_names) , 1 ] = 1

def vec2c(vec):
	for i,v in enumerate(vec):
		if v:
			return indices_char[i]
	return ""

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X, y,validation_split=0.2, callbacks=[WandbCallback()])

