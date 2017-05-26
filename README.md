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

### Install python

You can download python from https://www.python.org/downloads/.  There are more detailed instructions for windows installation at https://www.howtogeek.com/197947/how-to-install-python-on-windows/.  

The material should work with python 2 or 3.  If you don't know the difference, installing python 2 seems to lead to slightly less problems with the installation.  

### Clone this github repository
```
git clone https://github.com/lukas/ml-class.git
cd ml-class
```

If you get an error message here, most likely you don't have git installed.  Go to https://www.atlassian.com/git/tutorials/install-git for intructions on installing git.

### Install necessary pip libraries
```
pip install pandas
pip install scikit-learn
pip install tensorflow
pip install keras
```

### Common problems

The most common problem is an old version of python.  Its easy to have multiple versions of python installed at once and Macs in particular come with a default version of python that is too old to install tensorflow.

Try running:

```
python --version
```

If your version is less than 2.7.12, you have a version issue.  Try reinstalling python 2.


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

# Class Agenda

Order of presentation of files, if you want to follow along

## Scikit Class
- test-scikit.py
- load-data.py
- pandas-explore.py
- feature-extraction-1.py
- feature-extraction-2.py
- feature-extraction-3.py
- classifier.py
- test-algorithm-1.py
- test-algorithm-2.py
- test-algorithm-dummy.py
- test-algorithm-cross-validation.py
- test-algorithm-cross-validation-dummy.py
- custom-tokenizer.py
- pipeline.py
- pipeline-bigrams.py
- pipeline-bigrams-cross-validation.py
- feature-selection.py
- grid-search.py

## Keras Class

- perceptron.py
- keras-digits.py
- keras-scikit-learn.py
- keras-one-hot.py
- keras-perceptron-1.py
- keras-perceptron-2.py
- keras-perceptron-3.py
- keras-perceptron-4.py
- keras-perceptron-save.py
- keras-perceptron-load.py
- keras-mlp.py
- keras-dropout.py
- tensorflow-mult.py
- tensorflow-perceptron.py
- keras-cnn-1.py
- keras-cnn-2.py
- keras-deep-dream.py

- keras-vgg-inspect.py
- keras-resnet50-inspect.py
- keras-inception-inspect.py
