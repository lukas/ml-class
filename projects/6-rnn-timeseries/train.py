# Some of this code from the excellent https://machinelearningmastery.com
#
# The project here is to predict temperatures from the daily-min-temperatures.csv
# file.  This is the minimum temperature in Melbourne over a period of 10 years.
# You can find more data at https://github.com/jbrownlee/Datasets.
#
# This model with a SimpleRNN gets mean absolue error (mae) of around 9-10 degrees C.
#
# Can you get the average absolue error to below 1.85 degrees C?


import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import CuDNNLSTM, LSTM, SimpleRNN, Dropout
from keras.callbacks import LambdaCallback

import wandb
from wandb.keras import WandbCallback

import plotutil
from plotutil import PlotCallback

wandb.init()
config = wandb.config

# If repeated prediction is True, the green line in the wandb plot will correspond to
# using the past prediction as input to the next prediction (hard case).
# If repeated prediction is False, the green line in the wandb plot will correspond
# to make in a prediction off of ground truth data every time.
config.repeated_predictions = False
config.look_back = 20

df = pd.read_csv('daily-min-temperatures.csv')
data = df.Temp.astype('float32').values

# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-config.look_back-1):
        a = dataset[i:(i+config.look_back)]
        dataX.append(a)
        dataY.append(dataset[i + config.look_back])
    return np.array(dataX), np.array(dataY)
    
# normalize data to between 0 and 1
#max_val = max(data)
#min_val = min(data)
#data=(data-min_val)/(max_val-min_val)

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
model.add(SimpleRNN(1, input_shape=(config.look_back,1 )))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(trainX, trainY, epochs=1000, batch_size=1, validation_data=(testX, testY),  callbacks=[WandbCallback(), PlotCallback(trainX, trainY, testX, testY, config.look_back, config.repeated_predictions)])





