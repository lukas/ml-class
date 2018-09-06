import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, SimpleRNN, Dropout
from keras.callbacks import LambdaCallback

import wandb
from wandb.keras import WandbCallback

import plotutil
from plotutil import PlotCallback

wandb.init()
config = wandb.config

config.repeated_predictions = False
config.look_back = 20

def load_data(data_type="airline"):
    if data_type == "flu":
        df = pd.read_csv('flusearches.csv')
        data = df.flu.astype('float32').values
    elif data_type == "airline":
        df = pd.read_csv('international-airline-passengers.csv')
        data = df.passengers.astype('float32').values
    elif data_type == "sin":
        df = pd.read_csv('sin.csv')
        data = df.sin.astype('float32').values
    return data

# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-config.look_back-1):
        a = dataset[i:(i+config.look_back)]
        dataX.append(a)
        dataY.append(dataset[i + config.look_back])
    return np.array(dataX), np.array(dataY)

data = load_data()
    
# normalize data to between 0 and 1
max_val = max(data)
min_val = min(data)
data=(data-min_val)/(max_val-min_val)

# split into train and test sets
split = int(len(data) * 0.70)
train = data[:split]
test = data[split:]

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]

# create and fit the RNN
model = Sequential()
model.add(Flatten(input_shape=(config.look_back,1 )))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=10, validation_data=(testX, testY),  callbacks=[WandbCallback(), PlotCallback(trainX, trainY, testX, testY, config.look_back)])





