import os.path
import numpy as np
import pandas as pd
import wandb

if not os.path.isfile('sign-language/sign_mnist_train.csv'):
    print("""Can't find data file, please run the following command from this directory:
  curl https://storage.googleapis.com/wandb-production.appspot.com/mlclass/sign-language-data.tar.gz | tar xvz""")
    exit()

def load_train_data():
    df=pd.read_csv('sign-language/sign_mnist_train.csv')
    X = df.values[:,1:].reshape(-1,28,28)
    y = df.values[:,0]
    return X, y

def load_test_data():
    df=pd.read_csv('sign-language/sign_mnist_test.csv')
    X = df.values[:,1:].reshape(-1,28,28)
    y = df.values[:,0]
    return X, y

X,y = load_train_data()
test_X, test_y = load_test_data()
print(X.shape)
print(y.shape)
