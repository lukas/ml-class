import pandas as pd
import numpy as np
from keras.utils import to_categorical

import wandb
from wandb.wandb_keras import WandbKerasCallback
run = wandb.init()
config = run.config

df = pd.read_csv('../scikit/tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

# mapping from labels to numbers
mapping = {'Negative emotion':0,
    'No emotion toward brand or product':1,
    'Positive emotion':2,
    'I can\'t tell':3}
numeric_target = [mapping[t] for t in fixed_target]
num_labels = len(mapping)

# one hot encode outputs
labels = to_categorical(numeric_target)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=config.max_words)
tokenizer.fit_on_texts(fixed_text)
sequences = tokenizer.texts_to_sequences(fixed_text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=config.max_sequence_length)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(config.validation_split * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

# Load the word embedding
embeddings_index = {}
f = open('../scikit/glove/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Embedding, Input, Dense, Flatten, Conv1D
from keras.layers import MaxPooling1D, Dropout
from keras.models import Model

embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=config.max_sequence_length,
                            trainable=False)

sequence_input = Input(shape=(config.max_sequence_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Dropout(0.3)(embedded_sequences)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Dropout(0.3)(x)
#x = Conv1D(128, 5, activation='relu')(x)
#x = MaxPooling1D(5)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(num_labels, activation='softmax')(x)

model = Model(sequence_input, preds)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=config.epochs, batch_size=config.batch_size,
          callbacks=[WandbKerasCallback()])
