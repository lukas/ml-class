# Projects

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lukas/ml-class/master?urlpath=%2Flab)

These are specific bite-sized projects to learn an aspect of deep learning, starting from scratch. There is an associated video around 10 minutes long. I assume no background in ML but some proficiency in python. The projects are in order from beginner to more advanced, but feel free to skip around.

| Project                                         | Starter Code                                                                                                   | Video                                                                                  |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Build a simple image classifier for apparel     | [projects/1-fashion-mnist](https://github.com/lukas/ml-class/tree/master/projects/1-fashion-mnist)             | [Build your first machine learning model](https://www.youtube.com/watch?v=CbXj7091OWA) |
| Improve your image classifier                   | [projects/2-fashion-mnist-mlp](https://github.com/lukas/ml-class/tree/master/projects/2-fashion-mnist-mlp)     | [Multi-Layer Perceptrons](https://www.youtube.com/watch?v=GVKDa5hxUZE)                 |
| Build a convolutional image classifier          | [projects/3-fashion-mnist-cnn](https://github.com/lukas/ml-class/tree/master/projects/3-fashion-mnist-cnn)     | [Convolutional Neural Networks](https://www.youtube.com/watch?v=wzy8jI-duEQ)           |
| Build a denoising autoencoder                   | [projects/4-fashion-autoencoder](https://github.com/lukas/ml-class/tree/master/projects/4-fashion-autoencoder) | [Autoencoders](https://www.youtube.com/watch?v=6maH8Lh3pK4)                            |
| Build a text classifier with Scikit-Learn       | [projects/5-sentiment-analysis](https://github.com/lukas/ml-class/tree/master/projects/5-sentiment-analysis)   | [Sentiment Analysis](https://www.youtube.com/watch?v=qoyp8pBtCZ0)                      |
| Predict the weather with an RNN                 | [projects/6-rnn-timeseries](https://github.com/lukas/ml-class/tree/master/projects/6-rnn-timeseries)           | [Recurrent Neural Networks](https://www.youtube.com/watch?v=8lbGjKhrJOo)               |
| Build a text generator                          | [projects/7-text-generator](https://github.com/lukas/ml-class/tree/master/projects/7-text-generator)           | [Text Generation using LSTMs and GRUs](https://www.youtube.com/watch?v=4F69m3krMHw)    |
| Build a sentiment classifier on Amazon reviews. | [projects/8-text-classification](https://github.com/lukas/ml-class/tree/master/projects/8-text-classification) | [Text Classification using CNNs](https://www.youtube.com/watch?v=8YsZXTpFRO0)          |
|                                                 |                                                                                                                | [Hybrid LSTM/CNNs](https://www.youtube.com/watch?v=NysY9FN9Uac)                        |
|                                                 |                                                                                                                | [Seq2seq Models](https://www.youtube.com/watch?v=MqugtGD605k)                          |
|                                                 |                                                                                                                | [Transfer Learning](https://www.youtube.com/watch?v=vbhEnEbj3JM)                       |
|                                                 |                                                                                                                | [One Shot Learning](https://www.youtube.com/watch?v=H4MPIWX6ftE)                       |
|                                                 |                                                                                                                | [Speech Recognition](https://www.youtube.com/watch?v=Qf4YJcHXtcY)                      |
|                                                 |                                                                                                                | [Data Augmentation](https://www.youtube.com/watch?v=yYqAvlkRwUQ)                       |
|                                                 |                                                                                                                | [Batch Size and Learning Rate](https://www.youtube.com/watch?v=ZBVwnoVIvZk)            |

# More Projects: Benchmarks

If you have done all of the tutorial projects, we have a few more that don't have associated lessons yet!
You can learn by contributing to one of our collaborative [Benchmarks](https://www.wandb.com/benchmarks).

| Project                          | Link                                                      |
| -------------------------------- | --------------------------------------------------------- |
| Japanese handwriting recognition | https://app.wandb.ai/wandb/kmnist/benchmark               |
| Video prediction with cat GIFs   | https://app.wandb.ai/wandb/catz/benchmark                 |
| Drought detection from satellite | https://app.wandb.ai/wandb/droughtwatch/benchmark         |
| Visual game playing              | https://app.wandb.ai/wandb/witness/benchmark              |
| Image resolution enhancement     | https://app.wandb.ai/wandb/superres/benchmark             |
| Photo colorization               | https://app.wandb.ai/wandb/colorizer-applied-dl/benchmark |

# Getting Started

1. Clone this repository
2. Get the python libraries (run 'pip install -r requirements.txt')

You don't need a fancy computer to run most of the examples, but especially to do the later projects you may want to invest in a GPU.

# Slides
[O'Reilly 9.10.2019 - Using Keras to classify text using LSTMs](https://www.dropbox.com/s/sdep4yralq4exq3/Oreilly%20LSTM%20-%20Sept%2010.pdf?dl=0)

# Videos
[Introduction to Machine Learning](https://www.wandb.com/classes/intro/overview)

# Examples

In my in-person classes, I typically use a lot of the examples in the directory _examples_. This code is liable to change as I update things.

# Reusing the materials

Please feel free to use these materials for your own classes/projects etc. If you do that, I would love it if you sent me a message and let me know what you're up to.

### Windows

#### Git

Install git if you don't have it: https://git-scm.com/download/win

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

If don't see "Anaconda" in the output, search for "anaconda prompt" from the start menu and enter your command prompt this way. It's also best to use a virtual environment to keep your packages silo'ed. Do so with:

```
conda create -n ml-class python=3.6
activate ml-class
```

Whenever you start a new terminal, you will need to call `activate ml-class`.

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

You can download python from https://www.python.org/downloads/. There are more detailed instructions for windows installation at https://www.howtogeek.com/197947/how-to-install-python-on-windows/.

The material should work with python 2 or 3. _On Windows, you need to install thre 64 bit version of python 3.5 or 3.6 in order to install tensorflow_.

#### Clone this github repository

```
git clone https://github.com/lukas/ml-class.git
cd ml-class
```

If you get an error message here, most likely you don't have git installed. Go to https://www.atlassian.com/git/tutorials/install-git for intructions on installing git.

#### Install necessary pip libraries

```
pip install -r requirements.txt
```

### Reading material for people who haven't done a lot of programming

If you are uncomfortable opening up a terminal, I strongly recommend doing a quick tutorial before you take this class. Setting up your machine can be painful but once you're setup you can get a ton out of the class. I recommend getting started ahead of time.

If you're on Windows I recommend checking out http://thepythonguru.com/.

If you're on a Mac check out http://www.macworld.co.uk/how-to/mac/coding-with-python-on-mac-3635912/

If you're on linux, you're probably already reasonably well setup :).

If you run into trouble, the book Learn Python the Hard Way has installation steps in great detail: https://learnpythonthehardway.org/book/ex0.html. It also has a refresher on using a terminal in the appendix.

### Reading material for people who are comfortable with programming, but haven't done a lot of python

If you are comfortable opening up a terminal but want a python intro/refresher check out https://www.learnpython.org/ for a really nice introduction to Python.

### Suggestions for people who have done a lot of programming in python

A lot of people like to follow along with ipython or jupyter notebooks and I think that's great! It makes data exploration easier. I also really appreciate pull requests to make the code clearer.

If you've never used pandas or numpy - they are great tools and I use them heavily in my work and for this class. I assume no knlowedge of pandas and numpy but you may want to do some learning on your own. You can get a quick overview of pandas at http://pandas.pydata.org/pandas-docs/stable/10min.html. There is a great overview of numpy at https://docs.scipy.org/doc/numpy/user/quickstart.html.
