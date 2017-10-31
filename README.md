# Overview

These are materials I use for a class on machine learning with scikit-learn, keras and tensorflow.  Each file is a self contained unit that demonstrates a specific thing.  Downloading or cloning this repository before class is a great way to follow along.

# Setup

## Prerequisites

Getting your machine setup is essential to following along with a class.  I try really hard to make this class as easy to setup as possible, but setting up a machine for development can be challenging if you haven't done it before.  When in doubt, copy the error messages into Google!  All of the programs you need to install for this are extremely standard and for any issue you run into, hundreds of other people have encountered the same issue.

### Reading material for people who haven't done a lot of programming

If you are uncomfortable opening up a terminal, I strongly recommend doing a quick tutorial before you take this class.  Setting up your machine can be painful but once you're setup you can get a ton out of the class.  I recommend getting started ahead of time.

If you're on Windows I recommend checking out http://thepythonguru.com/.

If you're on a Mac check out http://www.macworld.co.uk/how-to/mac/coding-with-python-on-mac-3635912/

If you're on linux, you're probably already reasonably well setup :).

If you run into trouble, the book Learn Python the Hard Way has installation steps in great detail: https://learnpythonthehardway.org/book/ex0.html.  It also has a refresher on using a terminal in the appendix.

### Reading material for people who are comfortable with programming, but haven't done a lot of python

If you are comfortable opening up a terminal but want a python intro/refresher check out https://www.learnpython.org/ for a really nice introduction to Python.

### Suggestions for people who have done a lot of programming in python

A lot of people like to follow along with ipython or jupyter notebooks and I think that's great!  It makes data exploration easier.  I also really appreciate pull requests to make the code clearer.

If you've never used pandas or numpy - they are great tools and I use them heavily in my work and for this class.  I assume no knlowedge of pandas and numpy but you may want to do some learning on your own.  You can get a quick overview of pandas at http://pandas.pydata.org/pandas-docs/stable/10min.html.  There is a great overview of numpy at https://docs.scipy.org/doc/numpy/user/quickstart.html.

## Installation
Before the class, please run the following commands to install the prerequisite code.

### Windows

#### Git

Install git: https://git-scm.com/download/win

#### Anaconda

Install [anaconda](https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe)

Try running the following from the command prompt:

```
python --version
```

You should see something like

```
Python 3.6.1 :: Anaconda 4.4.0 (64-bit)
```

If don't see "Anaconda" in the output, search for "anaconda prompt" from the start menu and enter your command prompt this way.   It's also best to use a virtual environment to keep your packages silo'ed.  Do so with:

```
conda create -n ml-class python=3.6
activate ml-class
```

Whenever you start a new terminal, you will need to call `activate ml-class`.

#### Common problems

The most common problem is an old version of python.  Its easy to have multiple versions of python installed at once and Macs in particular come with a default version of python that is too old to install tensorflow.

Try running:

```
python --version
```

If your version is less than 2.7.12, you have a version issue.  Try reinstalling python 2.


#### Clone this github repository
```
git clone https://github.com/lukas/ml-class.git
cd ml-class
```

#### libraries

```
pip install wandb
conda install -c conda-forge scikit-learn
conda install -c conda-forge tensorflow
conda install -c conda-forge keras
```

### Linux and Mac OS X
#### Install python

You can download python from https://www.python.org/downloads/.  There are more detailed instructions for windows installation at https://www.howtogeek.com/197947/how-to-install-python-on-windows/.  

The material should work with python 2 or 3.  *On Windows, you need to install thre 64 bit version of python 3.5 or 3.6 in order to install tensorflow*.

#### Clone this github repository
```
git clone https://github.com/lukas/ml-class.git
cd ml-class
```

If you get an error message here, most likely you don't have git installed.  Go to https://www.atlassian.com/git/tutorials/install-git for intructions on installing git.

#### Install necessary pip libraries
```
pip install wandb
pip install pandas
pip install scikit-learn
pip install tensorflow
pip install keras
```

#### Install python libraries for optional material
```
pip install h5py
pip install flask
pip install scikit-image
pip install scipy
pip install pillow
```


## Check installation

To make sure your installation works go to the directory where this file is and run
```
python test-scikit.py
```

You should see the output "Scikit is installed!"

```
python test-keras.py
```

You should see the output "Using TensorFlow backend.  Keras is installed!"

## Download files before class

Please download the large files before the class if you can to save time and bandwidth

```
python keras-cifar-download.py
```

Also download VGG weights file from https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc

## Optional

If you want to download your own images for classification, one fun/easy way to do it is
install the Chrome extension bulk image downloader at http://www.talkapps.org/bulk-image-downloader


# Class Agenda

Order of presentation of files, if you want to follow along

## Scikit Class

### Introduction and Loading data
- scikit/load-data.py
- scikit/pandas-explore.py

### Feature Extraction
- scikit/feature-extraction-1.py
- scikit/feature-extraction-2.py
- scikit/feature-extraction-3.py

### Build your first classifier
- scikit/classifier.py


### Build another classifier
- scikit/classifier-svm.py


### Evaluating classifier performance
- scikit/test-algorithm-1.py
- scikit/test-algorithm-2.py
- scikit/test-algorithm-dummy.py
- scikit/test-algorithm-cross-validation.py
- scikit/test-algorithm-cross-validation-dummy.py
- scikit/custom-tokenizer.py

### Evaluating other algorithms and hyperparameters
- scikit/test-algorithm-cross-validation-hyper.py
- scikit/test-algorithm-cross-validation-rf.py
- scikit/test-algorithm-cross-validation-svm.py
- scikit/cross-validation-wandb.py

### Pipelines, Grid Search and Custom Features
- scikit/pipeline.py
- scikit/pipeline-bigrams.py
- scikit/pipeline-bigrams-cross-validation.py
- scikit/feature-selection.py
- scikit/grid-search.py
- scikit/pipeline-custom-features.py

### Model Save/Server
- scikit/pipeline-save.py
- scikit/pipeline-server.py

## Keras Class

- scikit/perceptron.py
- keras-perceptron/digits.py
- keras-perceptron/scikit-learn.py
- keras-perceptron/keras-one-hot.py
- keras-perceptron/log-loss.py
- keras-perceptron/perceptron-1.py
- keras-perceptron/perceptron-2.py
- keras-perceptron/perceptron-3.py
- keras-perceptron/perceptron-4.py
- keras-perceptron/perceptron-checkpoint.py
- keras-perceptron/perceptron-save.py
- keras-perceptron/perceptron-load.py
- keras-perceptron/perceptron-regression.py
- keras-mlp/mlp.py
- keras-mlp/dropout.py

### tensorflow
- tensorflow/mult.py
- tensorflow/perceptron.py
- tensorflow/cnn.py

### build your own nn puzzle
- keras-puzzle/keras-weights.py
- keras-puzzle/keras-weights-answer.py

### conv neural nets
- keras-cnn/convolution-demo.py
- keras-cnn/maxpool-demo.py
- keras-cnn/cnn-1.py
- keras-cnn/cnn-2.py
- keras-cnn/cnn-inspect.py

### deep dream
- keras-deep-dream/dream.py

### standard models
- keras-transfer/vgg-inspect.py
- keras-transfer/resnet50-inspect.py
- keras-transfer/inception-inspect.py

### smile data set (serving models)
- keras-smile/smile.py
- keras-smile/smile-generator.py
- keras-smile/smile-server-1.py
- keras-smile/smile-server-2.py
- keras-smile/smile-server-3.py

### cat v dog
- keras-transfer/dogcat-1.py
- keras-transfer/dogcat-generator.py
- keras-transfer/dogcat-bottleneck.py
- keras-transfer/dogcat-transfer.py
- keras-transfer/dogcat-finetune.py
- keras-transfer/dogcat-transfer-and-finetune.py

### Time series LSTM
- keras-lstm/lstm-time-series-output.py
- keras-lstm/lstm-time-series-stateful.py
- keras-lstm/lstm-time-series-timesteps.py
- keras-lstm/lstm-time-series-window.py


### LSTM for generating text
- keras-lstm/lstm-preprocess-text.py
- keras-lstm/lstm-generate-text-rnn.py
- keras-lstm/lstm-train-rnn.py
- keras-lstm/lstm-train-lstm.py
- keras-lstm/lstm-generate-text-lstm.py
- keras-lstm/lstm-generate-text-rnn.py
- keras-lstm/lstm-imdb-sentiment-cnn.py

### visualization
- keras-cnn/inspect-net.py

### Adversarial network
- keras-gan/gan.py
