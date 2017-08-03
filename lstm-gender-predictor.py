from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import __version__ as keras_version
import numpy as np


nEpochs = 10
weightsFileName = "gender_weights.h5"
with open("male.txt") as f:
    m_names = f.readlines()

with open("female.txt") as f:
    f_names = f.readlines()

mf_names = []

for f_name in f_names:
	if f_name in m_names:
		mf_names.append(f_name)

m_names = [m_name.lower() for m_name in m_names if not m_name in mf_names]
f_names = [f_name.lower() for f_name in f_names if not f_name in mf_names]


totalEntries = len(m_names) + len(f_names)
maxlen = len(max( m_names , key=len)) + len(max( f_names , key=len))

chars = set(  "".join(m_names) + "".join(f_names)  )
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X = np.zeros((totalEntries , maxlen, len(chars) ), dtype=np.bool)
y = np.zeros((totalEntries , 2 ), dtype=np.bool)

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
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

json_string = model.to_json()

with open("model.json", "w") as text_file:
    text_file.write(json_string)


if keras_version[0] == '1':
	model.fit(X, y, batch_size=16, nb_epoch=nEpochs)
else:
	model.fit(X, y, batch_size=16, epochs=nEpochs,validation_split=0.2)

model.save('gender.h5')

score = model.evaluate(X, y, batch_size=16)
print "score " , score
