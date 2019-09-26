import numpy as np
import pandas as pd
import wandb
import tensorflow as tf
from plotutil import PlotCallback

wandb.init()
config = wandb.config
config.repeated_predictions = False
config.batch_size = 40
config.look_back = 4
config.epochs = 500


def load_data(data_type="airline"):
    """read a CSV into a dataframe"""
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


def create_dataset(dataset):
    """convert an array of values into a dataset matrix"""
    dataX, dataY = [], []
    for i in range(len(dataset)-config.look_back-1):
        a = dataset[i:(i+config.look_back)]
        dataX.append(a)
        dataY.append(dataset[i + config.look_back])
    return np.array(dataX), np.array(dataY)


data = load_data("sin")

# normalize data to between 0 and 1
max_val = max(data)
min_val = min(data)
data = (data-min_val)/(max_val-min_val)

# split into train and test sets
split = int(len(data) * 0.70)
train = data[:split]
test = data[split-config.look_back-2:]

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

# Add channel dimension
trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]

# create and fit the RNN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(5, input_shape=(config.look_back, 1)))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss='mae', optimizer='rmsprop')
model.fit(trainX, trainY, epochs=config.epochs, batch_size=config.batch_size, validation_data=(testX, testY),  callbacks=[
          PlotCallback(trainX, trainY, testX, testY,
                       config.look_back, config.repeated_predictions),
          wandb.keras.WandbCallback(input_type="time")])
