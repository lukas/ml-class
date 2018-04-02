
import json
 
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import numpy as np

import pandas as pd
import numpy as np

import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config

config.max_words = 1000
config.max_length = 300
# Puts tweets into a data frame
df = pd.read_csv('tweets.csv')

target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text'].astype(str)

category_to_num = {"I can't tell": 0, "Negative emotion": 1, "Positive emotion": 2, "No emotion toward brand or product": 3}
target_num = [category_to_num[t] for t in target]
target_one_hot = np_utils.to_categorical(target_num)

tokenizer = Tokenizer(num_words=config.max_words)
tokenizer.fit_on_texts(list(text))
sequences = tokenizer.texts_to_sequences(list(text))
data = pad_sequences(sequences, maxlen=config.max_length)

train_data = data[:6000]
test_data = data[6000:]
train_target = target_one_hot[:6000]
test_target = target_one_hot[6000:]

model = Sequential()
model.add(Embedding(config.max_words, 128, input_length=300))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_target, callbacks=[WandbKerasCallback()], validation_data=(test_data, test_target))
