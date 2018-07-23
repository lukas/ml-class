import os.path
import numpy as np
import pandas as pd
import wandb

if not os.path.isfile('sign_mnist_train.csv'):
    raise Exception("Can't find data file, please go to https://www.kaggle.com/datamunge/sign-language-mnist and download sign-language-mnist.zip and then unzip in the local directory")

def load_train_data():
    df=pd.read_csv('sign_mnist_train.csv')
    X = df.values[:,1:].reshape(-1,28,28)
    y = df.values[:,0]
    return X, y

def load_test_data():
    df=pd.read_csv('sign_mnist_test.csv')
    X = df.values[:,1:].reshape(-1,28,28)
    y = df.values[:,0]
    return X, y

X,y = load_train_data()
test_X, test_y = load_test_data()
print(X.shape)
print(y.shape)
