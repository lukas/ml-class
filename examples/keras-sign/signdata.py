# Code to load the sign data

import os.path
import numpy as np
import pandas as pd
import wandb
import subprocess

if not os.path.isfile('sign-language/sign_mnist_train.csv'):
    print("Downloading signlanguage dataset...")
    subprocess.check_output("curl https://storage.googleapis.com/wandb-production.appspot.com/mlclass/sign-language-data.tar.gz | tar xvz", shell=True)

letters = "abcdefghijklmnopqrstuvwxyz"

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

